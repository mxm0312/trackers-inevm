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
		self.events = ['/events/{}/before_start',		#Вызывается как можно раньше после запуска контейнера и перед началом работы скриптов обработки данных
				'/events/{}/on_progress',		#Вызывается периодически по мере обработки данных в контейнере
				'/events/{}/on_error',	#Вызывается в случае возникновения ошибки в процессе обработки данных в контейнере
				'/events/{}/before_end']	#Вызывается после окончания обработки данных в контейнере и сохранением выходных файлов в папке OUT_DIR и перед остановкой (удалением) контейнера

	def get_short_id(self):
		'''Получение краткого id контейнера, как имя хоста
		'''
		self.short_id = socket.gethostname()

	def get_containers_from_docker(self)->dict:
		'''Получение списка активных контейнеров от docker
		Return: словарь вида full_id:name
		'''
		result = subprocess.check_output('docker ps --format "{{.ID}}: {{.Names}}" --no-trunc', shell=True).decode("utf-8").split('\n')
		containers = {}
		for line in result:
			if ': ' in line:
				key, value = line.split(': ')
				containers[key] = value
		return containers

	def get_id(self, full = True):
		'''Получение полного id контейнера
		'''
		self.get_short_id()
		if full:
			containers = self.get_containers_from_docker()
			for id_ in containers.keys():
				if self.short_id == id_[:12]:
					self.full_id = id_
					break
		
	def post(self, url, data = None):
		'''Шаблон post запроса
		Return: ответ на запрос
		'''
		#print(url)
		res = {}
		try:
			res = requests.post(url, data)
		except Exception as e:
			print("Error: " +str(e))
		finally:
			return res
		
	def post_status(self, status):
		'''Шаблон post запроса со статусом
		Return: ответ на запрос
		'''
		return self.post(urljoin(self.host_web,self.events[status].format(self.full_id)))
		
	def post_start(self):
		'''Шаблон post запроса со статусом before_start
		Return: ответ на запрос
		'''
		return self.post_status(0)
		
	def post_progress(self):
		'''Шаблон post запроса со статусом on_progress
		Return: ответ на запрос
		'''
		return self.post_status(1)
		
	def post_error(self):
		'''Шаблон post запроса со статусом on_error
		Return: ответ на запрос
		'''
		return self.post_status(2)
		
	def post_end(self):
		'''Шаблон post запроса со статусом before_end
		Return: ответ на запрос
		'''
		return self.post_status(3)

def get_id():
	'''Получение полного id контейнера
	'''
	full_id = ""
	containers = {}
	
	short_id = socket.gethostname()
	result = subprocess.check_output('docker ps --format "{{.ID}}: {{.Names}}" --no-trunc', shell=True).decode("utf-8").split('\n')
	for line in result:
		if ': ' in line:
			key, value = line.split(': ')
			containers[key] = value
	for id_ in containers.keys():
		if short_id == id_[:12]:
			full_id = id_
			break
	return full_id
