FROM public.ecr.aws/lambda/nodejs:24 AS build
WORKDIR /build
COPY package.json ./
COPY laya-ts ./laya-ts
RUN npm install --omit=optional
COPY src/ src/
RUN npx esbuild src/handler.ts --bundle --platform=node --format=esm --target=node24 \
      --outfile=handler.mjs \
      --external:onnxruntime-node

FROM public.ecr.aws/lambda/nodejs:24
WORKDIR ${LAMBDA_TASK_ROOT}
COPY model/ /opt/model/
COPY package.json ./
RUN npm install --omit=dev --omit=optional
COPY --from=build /build/handler.mjs ${LAMBDA_TASK_ROOT}/handler.mjs
CMD ["handler.handler"]
