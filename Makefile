redis:
	docker run -d --name redis-stack -v redis-data:/data -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
