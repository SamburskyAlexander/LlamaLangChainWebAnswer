# LlamaLangChainWebAnswer

Веб-сервис. Отвечает на запросы пользователя по базе знаний.

### Инструкция
1. Файлы с необходимыми данными (csv/pdf-формат) положить в директорию `docs`;
2. Файл с моделью положить в директорию `models`;
3. Переименовать файл с моделью в `llama-2-7b-chat.Q4_K_M.gguf` при необходимости;
4. `docker build -t container_name .`
5. `docker run −p 8080:8080 container_name`
