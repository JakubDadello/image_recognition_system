FROM industrial_defect_detector:5m4pykymscmvtymn

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.7.1 /lambda-adapter /opt/extensions/lambda-adapter

ENV PORT=3000
CMD ["python", "-m", "bentoml", "serve", ".", "--production", "--port", "3000"]