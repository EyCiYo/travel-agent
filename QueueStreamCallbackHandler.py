from langchain.callbacks.base import BaseCallbackHandler


class QueueCallback(BaseCallbackHandler):

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        print(token)
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()
