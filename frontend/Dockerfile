FROM node:18.17.0-alpine AS base

FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /usr/local/app

COPY . .

# ENV NODE_ENV production

RUN npm install

# RUN npm run build