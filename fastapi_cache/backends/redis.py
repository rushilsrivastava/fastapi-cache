from typing import Any, Optional, Tuple, Union

from redis.asyncio.client import Redis
from redis.asyncio.cluster import RedisCluster

from fastapi_cache.types import Backend


class RedisBackend(Backend):
    def __init__(
        self,
        redis: Union[
            "Redis[bytes]",
            "RedisCluster[bytes]",
            "Redis[str]",
            "RedisCluster[str]",
        ],
    ):
        self.redis = redis
        self.is_cluster: bool = isinstance(redis, RedisCluster)

    async def get_with_ttl(self, key: str) -> Tuple[int, Optional[bytes]]:
        async with self.redis.pipeline(transaction=not self.is_cluster) as pipe:
            return await pipe.ttl(key).get(key).execute()  # type: ignore[union-attr,no-any-return]

    async def get(self, key: str) -> bytes | Any | str | None:
        return await self.redis.get(key)  # type: ignore[union-attr]

    async def set(
        self, key: str, value: bytes, expire: Optional[int] = None
    ) -> None:
        await self.redis.set(key, value, ex=expire)  # type: ignore[union-attr]

    async def clear_namespace_non_block(
        self, namespace: str, count: int = 1000, batch_size: int = 1000
    ) -> int:
        lua_script = """
        local cursor = "0"
        local count = tonumber(ARGV[2])
        local total_deleted = 0
        local batch_size = tonumber(ARGV[3])  -- Maximum keys to delete at once

        repeat
            local result = redis.call("SCAN", cursor, "MATCH", ARGV[1] .. ":*", "COUNT", count)
            cursor = tostring(result[1])
            local keys = result[2]

            if #keys > 0 then
                -- Delete in batches if we have too many keys
                for i = 1, #keys, batch_size do
                    local batch_end = math.min(i + batch_size - 1, #keys)
                    local batch = {}

                    -- Extract batch of keys
                    for j = i, batch_end do
                        table.insert(batch, keys[j])
                    end

                    -- Delete batch
                    if #batch > 0 then
                        redis.call("DEL", unpack(batch))
                        total_deleted = total_deleted + #batch
                    end
                end
            end

        until cursor == "0"

        return total_deleted
        """
        return await self.redis.eval(
            lua_script, 0, namespace, count, batch_size
        )  # type: ignore[union-attr,no-any-return]

    async def clear(
        self, namespace: Optional[str] = None, key: Optional[str] = None
    ) -> int:
        if namespace:
            lua = f"for i, name in ipairs(redis.call('KEYS', '{namespace}:*')) do redis.call('DEL', name); end"
            return await self.redis.eval(lua, numkeys=0)  # type: ignore[union-attr,no-any-return]
        elif key:
            return await self.redis.delete(key)  # type: ignore[union-attr]
        return 0

    async def close(self) -> None:
        return await self.redis.aclose()  # type: ignore[union-attr, no-any-return]
