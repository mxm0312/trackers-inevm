import socket
import subprocess
import requests
from urllib.parse import urljoin

class ContainerStatus():

	def __init__(self, host_web):
		self.host_web = host_web
		self.short_id = None
		self.full_id = None
		self.get_id()
		self.events = ['/events/{}/before_start',	#Вызывается как можно раньше после запуска контейнера и перед началом работы скриптов обработки данных
				'/events/{}/on_progress',			#Вызывается периодически по мере обработки данных в контейнере
				'/events/{}/on_error',				#Вызывается в случае возникновения ошибки в процессе обработки данных в контейнере
				'/events/{}/before_end']			#Вызывается после окончания обработки данных в контейнере и сохранением выходных файлов в папке OUT_DIR и перед остановкой (удалением) контейнера

	def get_short_id(self):
		pass

	def get_containers_from_docker(self)->dict:
		pass

	def get_id(self, full:bool = True):
		pass
		
	def post(self, url:str, data:dict = {}):
		pass
		
	def post_status(self, status:int, data:dict = None):
		pass
		
	def post_start(self,data:dict = None):
		pass
		
	def post_progress(self,data = None):
		pass
		
	def post_error(self,data = None):
		pass
		
	def post_end(self,data = None):
		pass

def get_id()->str:
	return 0
