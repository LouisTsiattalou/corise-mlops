FROM python:3.9

WORKDIR /project

# Copy over requirements.txt separately to use cache for pip install
# (instead of reinstalling with every build)
ADD ./requirements.txt /project/requirements.txt
RUN pip install --upgrade -r requirements.txt

# Copy the rest of the files
ADD . /project
WORKDIR /project/app

# Start uvicorn server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
