"""Task 09：AsyncEngine——把同步的 ``EngineCore`` 包装成可流式、可取消的 asyncio 服务层。

要解决的三个问题（docs/02 §9、§10）：

1. **``EngineCore.step()`` 是阻塞的**（一次前向 = 一个 token）。直接在协程里调会把事件循环焊死，
   SSE 一个字都发不出去。所以阻塞的前向用 ``asyncio.to_thread`` 卸载到线程，事件循环只做投递。
2. **``EngineCore`` 不是线程安全的**（registry / scheduler / 块池都是裸 dict）。HTTP handler 在
   事件循环线程，前向在 worker 线程，两边同时碰 core 就是数据竞争。解决办法是**单写者纪律**：
   所有对 core 的读写都只在后台 ``_run_loop`` 这一个协程里发生，外部一律通过命令队列投递意图。
3. **客户端断连必须回收 KV**（docs/02 §10）。断连时 Starlette 会关闭响应体生成器，我们在
   ``AsyncStream`` 的 ``finally`` 里补一次 abort，落到 ``EngineCore.cancel`` —— 而 Task 08 已
   保证 cancel 立刻 ``free_table`` 归还物理块。

本模块**不改动 EngineCore 一行代码**：它只依赖 core 的 ``submit / step / cancel / get_request /
registry.active()`` 五个既有方法。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, TYPE_CHECKING

from liteinfer.sampling.params import SamplingParams

if TYPE_CHECKING:  # 只作类型标注用，避免与 engine.core 形成运行时导入环
    from liteinfer.engine.core import EngineCore

logger = logging.getLogger("liteinfer.engine.async_engine")


@dataclass(frozen=True)
class StreamChunk:
    """流式输出的最小单元，语义对齐 OpenAI chunk 的 delta（增量而非累积）。

    - ``text`` 是**本次增量**文本（通常一个 token 解码出来的片段），不是从头累积的全文；
    - ``finished`` 为 True 表示这是该请求的最后一个 chunk；
    - ``finish_reason`` 只在 ``finished`` 为 True 时有意义（``"length"`` / ``"eos"`` /
      ``"cancelled"`` / ``"error"``）；
    - EOS 直接命中时 ``token_id`` 为 None（该 step 没有产出新 token），此时 ``text`` 为空串，
      这个 chunk 只是用来把 ``finish_reason`` 送出去。
    - ``index`` 是本 chunk 在**本请求**内的序号（从 0 计），便于前端/demo 观察进度。
    """

    request_id: str
    token_id: Optional[int]
    text: str
    finished: bool
    finish_reason: Optional[str]
    index: int = 0


@dataclass
class _Command:
    """引擎循环收到的内部命令。

    为什么不直接调 core：core 的所有访问必须串行化（见模块 docstring 第 2 点）。
    ``future`` 用于把结果（request_id 或完成信号）交还给发起方。
    """

    kind: str  # "submit" | "abort"
    request_id: Optional[str] = None
    prompt: Optional[str] = None
    params: Optional[SamplingParams] = None
    future: Optional[asyncio.Future] = None


class AsyncStream:
    """单个请求的输出流（async iterator）。

    为什么单独成类而不是让 ``AsyncEngine.generate`` 直接是 async generator：
    流式 HTTP 响应是"外层生成器（SSE）套内层生成器（本类）"，若内层靠 GC 兜底关闭，
    ``finally`` 的执行时机不确定，断连回收就不可验证。把流做成对象后，
    外层可以显式 ``await it.aclose()``，清理时机确定。
    """

    def __init__(self, engine: "AsyncEngine", request_id: str) -> None:
        self._engine = engine
        self.request_id = request_id
        self._completed = False

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        # 每次调用返回一个新的迭代器；同一条流只应被迭代一次（多个消费者会互相抢队列）
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[StreamChunk]:
        queue = self._engine._queue_for(self.request_id)
        try:
            while True:
                chunk = await queue.get()
                yield chunk
                if chunk.finished:
                    self._completed = True
                    return
        finally:
            # 正常结束（看到 finished chunk）时无需 abort；被提前关闭（客户端断连、
            # 消费方 break）时必须补一次取消，否则 KV 块会挂到进程结束。
            if not self._completed:
                self._engine.abort_nowait(self.request_id)
            self._engine._forget(self.request_id)


class AsyncEngine:
    """``EngineCore`` 的 asyncio 外壳：提交、流式消费、取消、关停。

    线程模型（重要）：
    - 后台 ``_run_loop`` 协程是**唯一**访问 ``EngineCore`` 的地方；
    - ``step()`` 通过 ``asyncio.to_thread`` 在 worker 线程执行，同一时刻最多一个 step 在飞，
      因此不需要给 core 加锁；
    - 外部（HTTP handler / demo）只碰命令队列和自己的输出队列，不直接碰 core。
    """

    def __init__(self, core: "EngineCore") -> None:
        self._core = core
        # 命令队列：把外部意图排队交给引擎循环串行执行
        self._cmd: asyncio.Queue[_Command] = asyncio.Queue()
        # 每请求一条输出队列（无界：token 产出速度本来就受前向限制，不需要在此处背压）
        self._out: dict[str, asyncio.Queue[StreamChunk]] = {}
        # 每个请求已投递的 chunk 数，用于填 StreamChunk.index
        self._counts: dict[str, int] = {}
        self._task: Optional[asyncio.Task] = None

    # ---- 观测 ----

    @property
    def core(self) -> "EngineCore":
        """被包装的同步引擎（只读用途：查状态/统计；不要从外部推进它）。"""
        return self._core

    @property
    def pending_streams(self) -> int:
        """尚未消费完的流的数量。"""
        return len(self._out)

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ---- 生命周期 ----

    async def start(self) -> None:
        """启动后台引擎循环（幂等；重复调用不会起第二个循环）。"""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._run_loop(), name="liteinfer-async-engine"
            )

    async def shutdown(self) -> None:
        """停掉引擎循环，并把所有在飞请求取消掉（回收它们的 KV 块）。

        取消在飞的这一步是必须的：只停循环会让请求永远留在 core 的 registry 里，
        物理块也就一直占着。
        """
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._finish_all_streams("cancelled")

    # ---- 对外 API ----

    async def submit(
        self, prompt: str, params: Optional[SamplingParams] = None
    ) -> str:
        """提交请求，返回 request_id（此时还没有任何 token 产出）。

        request_id 由 ``EngineCore.submit`` 生成，而它只能在引擎循环里被调用，
        所以这里用一个 future 把 id 交回来——代价是最多等一个 step，换来零竞态。
        ``EngineCore.submit`` 抛出的校验错误（如 prompt 超过 token budget）会原样传播。
        """
        await self.start()
        future = asyncio.get_running_loop().create_future()
        self._cmd.put_nowait(
            _Command("submit", prompt=prompt, params=params, future=future)
        )
        return await future

    def stream(self, request_id: str) -> AsyncStream:
        """拿到某请求的输出流。必须先 ``submit``。"""
        if request_id not in self._out:
            raise KeyError(f"未知或已结束的 request_id: {request_id!r}")
        return AsyncStream(self, request_id)

    async def generate(
        self, prompt: str, params: Optional[SamplingParams] = None
    ) -> AsyncStream:
        """提交并返回它的输出流（``submit`` + ``stream`` 的便捷组合）。

        返回的是 :class:`AsyncStream` 而不是 async generator，用法：

        .. code-block:: python

            stream = await engine.generate(prompt, params)
            async for chunk in stream:
                print(chunk.text, end="")
        """
        request_id = await self.submit(prompt, params)
        return self.stream(request_id)

    async def abort(self, request_id: str) -> None:
        """取消请求并**等待**取消真正生效（HTTP 取消端点用这条，保证响应时已回收）。"""
        if self.is_running():
            future = asyncio.get_running_loop().create_future()
            self._cmd.put_nowait(
                _Command("abort", request_id=request_id, future=future)
            )
            await future
        else:
            # 循环没在跑 => 不会有 step 在飞，直接取消是安全的
            self._cancel_now(request_id)

    def abort_nowait(self, request_id: str) -> None:
        """请求取消但**不等待**（用于生成器被关闭的清理路径）。

        为什么不能在这里 await：清理路径可能正跑在事件循环关闭/生成器 finalizer 里，
        此时 await 一个由引擎循环 resolve 的 future 有可能永远等不到。
        "投递命令 + 循环下一轮必执行"已经足够保证取消发生。
        """
        if self.is_running():
            self._cmd.put_nowait(_Command("abort", request_id=request_id))
        else:
            self._cancel_now(request_id)

    # ---- 供 AsyncStream 使用的内部访问器（同一模块内的友元访问） ----

    def _queue_for(self, request_id: str) -> "asyncio.Queue[StreamChunk]":
        if request_id not in self._out:
            raise KeyError(f"未知或已结束的 request_id: {request_id!r}")
        return self._out[request_id]

    def _forget(self, request_id: str) -> None:
        self._out.pop(request_id, None)
        self._counts.pop(request_id, None)

    # ---- 引擎循环 ----

    async def _run_loop(self) -> None:
        """后台主循环：drain 命令 → 推进一步 → 投递 token。

        命令只在"当前没有 step 在飞"时被处理，这是单写者纪律成立的地方。
        """
        logger.debug("AsyncEngine 引擎循环启动")
        try:
            while True:
                self._drain_commands()
                if self._has_work():
                    try:
                        results = await asyncio.to_thread(self._core.step)
                    except Exception:
                        # step 失败（例如块池耗尽）不能让循环静默死掉：取消全部在飞请求，
                        # 让 registry 清空，循环回到"等命令"的空闲态，引擎仍可为新请求服务。
                        logger.exception("引擎 step 失败，取消全部在飞请求")
                        self._finish_all_streams("error")
                        continue
                    self._dispatch(results)
                    continue
                # 空闲：阻塞等命令，避免空转烧 CPU
                self._handle(await self._cmd.get())
        except asyncio.CancelledError:
            logger.debug("AsyncEngine 引擎循环被取消")
            raise
        finally:
            self._finish_all_streams("cancelled")

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self._cmd.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._handle(cmd)

    def _has_work(self) -> bool:
        return bool(self._core.registry.active())

    def _handle(self, cmd: _Command) -> None:
        if cmd.kind == "submit":
            try:
                request_id = self._core.submit(cmd.prompt, cmd.params)
            except Exception as exc:  # 校验失败（如 prompt 超长）要回传给调用方
                if cmd.future is not None and not cmd.future.done():
                    cmd.future.set_exception(exc)
                return
            self._out[request_id] = asyncio.Queue()
            self._counts[request_id] = 0
            if cmd.future is not None and not cmd.future.done():
                cmd.future.set_result(request_id)
            return

        if cmd.kind == "abort":
            try:
                self._cancel_now(cmd.request_id or "")
            except Exception as exc:  # 取消失败也要把 future 结算掉，否则调用方永久挂起
                if cmd.future is not None and not cmd.future.done():
                    cmd.future.set_exception(exc)
                return
            if cmd.future is not None and not cmd.future.done():
                cmd.future.set_result(None)
            return

        logger.warning("忽略未知命令: %s", cmd.kind)

    def _dispatch(self, results: Any) -> None:
        """把 ``EngineCore.step()`` 的逐步结果投递到各请求的输出队列。"""
        for result in results:
            queue = self._out.get(result.request_id)
            if queue is None:  # 该请求已被取消并遗忘，丢弃即可
                continue
            index = self._counts.get(result.request_id, 0)
            self._counts[result.request_id] = index + 1
            queue.put_nowait(
                StreamChunk(
                    request_id=result.request_id,
                    token_id=result.token_id,
                    text=result.token_text,
                    finished=result.finished,
                    finish_reason=result.finish_reason,
                    index=index,
                )
            )

    def _cancel_now(self, request_id: str) -> None:
        """立刻（在引擎循环线程内）取消一个请求，并给它的流补一个终态 chunk。"""
        if not request_id:
            return
        try:
            request = self._core.get_request(request_id)
        except KeyError:
            logger.debug("取消未知请求 %s（可能已结束）", request_id)
            return
        was_terminal = request.is_terminal
        if not was_terminal:
            # Task 08：cancel 会立刻 free_table 归还物理块
            self._core.cancel(request_id)
        queue = self._out.get(request_id)
        if queue is not None and not was_terminal:
            queue.put_nowait(
                StreamChunk(
                    request_id=request_id,
                    token_id=None,
                    text="",
                    finished=True,
                    finish_reason="cancelled",
                    index=self._counts.get(request_id, 0),
                )
            )

    def _finish_all_streams(self, reason: str) -> None:
        """关停/异常路径：取消所有在飞请求并给每条流补终态 chunk（防止消费方永久挂起）。"""
        for request_id in list(self._out):
            self._cancel_now(request_id)
            queue = self._out.get(request_id)
            if queue is None:
                continue
            queue.put_nowait(
                StreamChunk(
                    request_id=request_id,
                    token_id=None,
                    text="",
                    finished=True,
                    finish_reason=reason,
                    index=self._counts.get(request_id, 0),
                )
            )
