from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):
    @task
    def predict(self):
        self.client.post("/predict", json={"text": "EPI = Echo planar imaging ."})

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)  # wait time between tasks in seconds
