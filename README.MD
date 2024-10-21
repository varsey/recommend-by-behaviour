# Рекомендации пользователям магазина на основе их поведения

Для запуска в линукс:

- `sudo apt-get update && sudo apt-get install build-essential`
- создайте вирутальное окружение на основе python3.11:
  - `python3 -m venv venv`
  - `source venv/bin/activate`
  - `pip install -r requirements.txt`
- убедитесь что файлы cat.csv (каталог товаров) и sample_actions.pq (дейсвтия  находятся в /data
- `python3 main.py`
- результирующий файл сохранится submit.csv в директории /data

# Запуск в docker

`docker build -t kokos`

`docker run -v ${PWD}/data:/opt/app/data -t kokos`

# Слайды с основными идеями

[Ссылка](https://docs.google.com/presentation/d/1pEIq6Hy2KbxS-Kpnqr3IXrhwQXEVySARLTcAv84Orjk/edit?usp=sharing)
