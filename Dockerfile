FROM python:3.11

WORKDIR /app
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
ADD ./static/sourcecode /app
ENV PYTHONPATH=/app/static/sourcecode
ENTRYPOINT ["python"]
CMD [ \
  "main.py", \
  "--enrollment", "opendata/userEnrollment-00000.tsv", \
  "--notes_path", "opendata/notes-00000.tsv", \
  "--ratings_path", "opendata/ratings-00000.tsv", \
  "--note_status_history_path", "opendata/noteStatusHistory-00000.tsv" \
]
