from config import agent
from vanna.servers.fastapi import VannaFastAPIServer

server = VannaFastAPIServer(agent)
server.run()
