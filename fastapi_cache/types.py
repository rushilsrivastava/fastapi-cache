import abc
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, Union

from starlette.requests import Request
from starlette.responses import Response
from typing_extensions import Protocol

_Func = Callable[..., Any]


class KeyBuilder(Protocol):
    def __call__(
        self,
        __function: _Func,
        __namespace: str = ...,
        *,
        request: Optional[Request] = ...,
        response: Optional[Response] = ...,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Union[Awaitable[str], str]: ...


class NamespaceBuilder(Protocol):
    def __call__(
        self,
        function: _Func,
        namespace: str = ...,
        *,
        kwargs: Dict[str, Any],
    ) -> str: ...


class Backend(abc.ABC):
    @abc.abstractmethod
    async def get_with_ttl(self, key: str) -> Tuple[int, Optional[bytes]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def get(self, key: str) -> bytes | str | None:
        raise NotImplementedError

    @abc.abstractmethod
    async def set(
        self, key: str, value: bytes, expire: Optional[int] = None
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def clear_namespace_non_block(
        self, namespace: str, count: int = 1000, batch_size: int = 1000
    ) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    async def clear(
        self, namespace: Optional[str] = None, key: Optional[str] = None
    ) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self) -> bool | None:
        raise NotImplementedError
