FROM node:22-alpine AS build

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
ARG NEXT_PUBLIC_RAG_API_URL
ENV NEXT_PUBLIC_RAG_API_URL=$NEXT_PUBLIC_RAG_API_URL
RUN npm run build

FROM node:22-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app ./
EXPOSE 3000
CMD ["npm", "run", "start", "--", "--host", "0.0.0.0", "--port", "3000"]
