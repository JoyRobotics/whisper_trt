import redis
import asyncio
import aioredis

class BaseRedis:
    _instances = {}

    def __new__(cls, host='127.0.0.1', port=6379, db=0, password='diablo_robot'):
        instance_key = f"{host}:{port}:{db}"
        if instance_key not in cls._instances:
            cls._instances[instance_key] = super(BaseRedis, cls).__new__(cls)
            cls._instances[instance_key]._initialized = False
        return cls._instances[instance_key]

    def __init__(self, host='127.0.0.1', port=6379, db=0, password='diablo_robot'):
        if not self._initialized:
            self.redis_url = f'redis://{host}:{port}'
            self.db = db
            self.password = password
            self.redis = None
            self._initialized = True

    async def ainitialize(self):
        #if not self.redis:
        self.redis = await aioredis.from_url(
            self.redis_url,
            db=self.db,
            password=self.password,
            encoding='utf-8',
            max_connections=10,
            decode_responses=True
        )

    def initialize(self):
        #if not self.redis:
        self.redis = redis.from_url(
            self.redis_url,
            db=self.db,
            password=self.password,
            encoding='utf-8',
            max_connections=10,
            decode_responses=True
        )

    def get_instance(self):
        #if not self.redis:
        self.initialize()
        return self.redis

    async def aget_instance(self):
        #if not self.redis:
        await self.ainitialize()
        return self.redis