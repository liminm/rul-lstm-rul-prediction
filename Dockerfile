FROM node:20-alpine AS frontend

WORKDIR /frontend

COPY sensor-dashboard/package*.json ./
RUN npm ci

COPY sensor-dashboard/ ./
RUN npm run build


FROM python:3.11-slim

WORKDIR /app

COPY requirements.infer.txt /app/requirements.infer.txt
RUN pip install --no-cache-dir -r /app/requirements.infer.txt

COPY . /app

RUN rm -rf /app/static
RUN mkdir -p /app/static

COPY --from=frontend /frontend/dist /app/static

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
