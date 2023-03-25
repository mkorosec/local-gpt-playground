redis:
	docker run -d --name redis-stack -v redis-data:/data -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

clear-redis-data:	
	docker stop redis-stack && docker rm redis-stack && docker volume rm redis-data

embed-json:
	python embed-gtr-5-to-json/embed.py test

embed-redis:
	python embed-gtr-5-to-redis/embed-redis.py test

install-npm-playground:
	npm install gpt-3-encoder
	npm install tsx

