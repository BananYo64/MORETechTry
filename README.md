# Установка и запуск проекта

## 1. Установка Apache
Установите Apache (пример для Ubuntu/Debian):
```bash
sudo apt update
sudo apt install apache2 -y
```

## 2. Размещение статических файлов
Скопируйте содержимое папки `site` в директорию:
```bash
sudo cp -r site/* /var/www/html/
```

После этого сайт будет доступен по адресу:  
[http://localhost](http://localhost)

## 3. Установка зависимостей
Установите необходимые Python-зависимости:
```bash
pip install -r requirements.txt
```

## 4. Запуск сервера
Запустите backend-сервер на **Uvicorn**:
```bash
uvicorn chatServer:app --host 0.0.0.0 --port 8000
```

## 5. Доступ к проекту
- Фронтенд: [http://localhost](http://localhost)  
- Backend: [http://localhost:8000](http://localhost:8000)
