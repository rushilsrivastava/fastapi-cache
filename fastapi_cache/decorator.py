import hashlib
import logging
import sys
from functools import wraps
from inspect import Parameter, Signature, isawaitable, iscoroutinefunction
from typing import (
    Awaitable,
    Callable,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

from fastapi.concurrency import run_in_threadpool
from fastapi.dependencies.utils import (
    get_typed_return_annotation,
    get_typed_signature,
)
from starlette.requests import Request
from starlette.responses import Response
from starlette.status import HTTP_304_NOT_MODIFIED

from fastapi_cache import FastAPICache
from fastapi_cache.coder import Coder
from fastapi_cache.types import KeyBuilder, NamespaceBuilder

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
P = ParamSpec("P")
R = TypeVar("R")


def _augment_signature(signature: Signature, *extra: Parameter) -> Signature:
    if not extra:
        return signature

    parameters = list(signature.parameters.values())
    variadic_keyword_params: List[Parameter] = []
    while parameters and parameters[-1].kind is Parameter.VAR_KEYWORD:
        variadic_keyword_params.append(parameters.pop())

    return signature.replace(
        parameters=[*parameters, *extra, *variadic_keyword_params]
    )


def _locate_param(
    sig: Signature, dep: Parameter, to_inject: List[Parameter]
) -> Parameter:
    """Locate an existing parameter in the decorated endpoint

    If not found, returns the injectable parameter, and adds it to the to_inject list.

    """
    param = next(
        (p for p in sig.parameters.values() if p.annotation is dep.annotation),
        None,
    )
    if param is None:
        to_inject.append(dep)
        param = dep
    return param


def _uncacheable(request: Optional[Request]) -> bool:
    """Determine if this request should not be cached

    Returns true if:
    - Caching has been disabled globally
    - This is not a GET request
    - The request has a Cache-Control header with a value of "no-store" or "no-cache"

    """
    if not FastAPICache.get_enable():
        return True
    if request is None:
        return False
    if request.method != "GET":
        return True
    return request.headers.get("Cache-Control") in ("no-store", "no-cache")


def cache(
    expire: Optional[int] = None,
    coder: Optional[Type[Coder]] = None,
    key_builder: Optional[KeyBuilder] = None,
    namespace: str = "",
    namespace_builder: Optional[NamespaceBuilder] = None,
    injected_dependency_namespace: str = "__fastapi_cache",
    private: bool = False,
    client_expire: Optional[int] = None,
) -> Callable[
    [Callable[P, Awaitable[R]]], Callable[P, Awaitable[Union[R, Response]]]
]:
    """
    cache all function
    :param namespace:
    :param expire:
    :param coder:
    :param key_builder:
    :param private:
    :param client_expire:
    :param injected_dependency_namespace:
    :param namespace_builder:

    :return:
    """

    injected_request = Parameter(
        name=f"{injected_dependency_namespace}_request",
        annotation=Request,
        kind=Parameter.KEYWORD_ONLY,
    )
    injected_response = Parameter(
        name=f"{injected_dependency_namespace}_response",
        annotation=Response,
        kind=Parameter.KEYWORD_ONLY,
    )

    def wrapper(
        func: Callable[P, Awaitable[R]],
    ) -> Callable[P, Awaitable[Union[R, Response]]]:
        # get_typed_signature ensures that any forward references are resolved first
        wrapped_signature = get_typed_signature(func)
        to_inject: List[Parameter] = []
        request_param = _locate_param(
            wrapped_signature, injected_request, to_inject
        )
        response_param = _locate_param(
            wrapped_signature, injected_response, to_inject
        )
        return_type = get_typed_return_annotation(func)

        @wraps(func)
        async def inner(
            *args: P.args, **kwargs: P.kwargs
        ) -> Union[R, Response]:
            nonlocal coder
            nonlocal expire
            nonlocal key_builder
            nonlocal client_expire
            nonlocal private
            nonlocal namespace

            async def ensure_async_func(*args: P.args, **kwargs: P.kwargs) -> R:
                """Run cached sync functions in thread pool just like FastAPI."""
                # if the wrapped function does NOT have request or response in
                # its function signature, make sure we don't pass them in as
                # keyword arguments
                kwargs.pop(injected_request.name, None)
                kwargs.pop(injected_response.name, None)

                if iscoroutinefunction(func):
                    # async, return as is.
                    # unintuitively, we have to await once here, so that caller
                    # does not have to await twice. See
                    # https://stackoverflow.com/a/59268198/532513
                    return await func(*args, **kwargs)
                else:
                    # sync, wrap in thread and return async
                    # see above why we have to await even although caller also awaits.
                    return await run_in_threadpool(func, *args, **kwargs)  # type: ignore[arg-type]

            copy_kwargs = kwargs.copy()
            request: Optional[Request] = copy_kwargs.pop(
                request_param.name, None
            )  # type: ignore[assignment]
            response: Optional[Response] = copy_kwargs.pop(
                response_param.name, None
            )  # type: ignore[assignment]

            if _uncacheable(request):
                return await ensure_async_func(*args, **kwargs)

            prefix = FastAPICache.get_prefix()
            coder = coder or FastAPICache.get_coder()
            expire = expire or FastAPICache.get_expire()
            client_expire = (
                client_expire if client_expire or client_expire == 0 else expire
            )
            key_builder = key_builder or FastAPICache.get_key_builder()
            backend = FastAPICache.get_backend()
            cache_status_header = FastAPICache.get_cache_status_header()
            local_namespace = namespace

            local_namespace = (
                namespace_builder(
                    func,
                    namespace=local_namespace,
                    kwargs=copy_kwargs,
                )
                if namespace_builder
                else local_namespace
            )

            cache_key = key_builder(
                func,
                f"{prefix}:{local_namespace}",
                request=request,
                response=response,
                args=coder.encode(args),
                kwargs=coder.encode(copy_kwargs),
            )
            if isawaitable(cache_key):
                cache_key = await cache_key
            assert isinstance(cache_key, str)  # noqa: S101  # assertion is a type guard

            try:
                cached = await backend.get(cache_key)
            except Exception:
                logger.warning(
                    f"Error retrieving cache key '{cache_key}' from backend:",
                    exc_info=True,
                )
                cached = None  # noqa: F841

            # Determine cache-control value
            cache_control = ""

            if client_expire == 0:
                cache_control += "no-cache, "
            else:
                cache_control += f"max-age={client_expire}, "

            if private:
                cache_control += "private, "

            cache_control = cache_control[:-2]

            if cached is None:  # cache miss
                result = await ensure_async_func(*args, **kwargs)
                to_cache = coder.encode(result)

                try:
                    await backend.set(cache_key, to_cache, expire)
                except Exception:
                    logger.warning(
                        f"Error setting cache key '{cache_key}' in backend:",
                        exc_info=True,
                    )

                if response:
                    if isinstance(to_cache, bytes):
                        hash_key = hashlib.md5(to_cache).hexdigest()  # noqa: S324
                    else:
                        hash_key = hashlib.md5(  # noqa: S324
                            str(to_cache).encode()
                        ).hexdigest()
                    etag = f"W/{hash_key}"
                    response.headers.update(
                        {
                            "Cache-Control": cache_control,
                            "ETag": etag,
                            cache_status_header: "MISS",
                        }
                    )

            else:  # cache hit
                if response:
                    if isinstance(cached, bytes):
                        hash_key = hashlib.md5(cached).hexdigest()  # noqa: S324
                    else:
                        hash_key = hashlib.md5(str(cached).encode()).hexdigest()  # noqa: S324
                    etag = f"W/{hash_key}"
                    response.headers.update(
                        {
                            "Cache-Control": cache_control,
                            "ETag": etag,
                            cache_status_header: "HIT",
                        }
                    )

                    if_none_match = request and request.headers.get(
                        "if-none-match"
                    )
                    if if_none_match == etag:
                        response.status_code = HTTP_304_NOT_MODIFIED
                        return response

                result = cast(
                    R, coder.decode_as_type(cached, type_=return_type)
                )

            # Close the backend
            try:
                await backend.close()
            except Exception:
                logger.warning(
                    "Error closing backend:",
                    exc_info=True,
                )

            return result

        inner.__signature__ = _augment_signature(wrapped_signature, *to_inject)  # type: ignore[attr-defined]

        return inner

    return wrapper
