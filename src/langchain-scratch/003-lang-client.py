from langserve import RemoteRunnable

if __name__ == "__main__":
    client = RemoteRunnable('http://localhost:8000/chain')
    output = client.stream(
        {
            "role": "whore",
            "drink": "coffee"
        }
    )
    for chunk in output:
        print(chunk,end='',flush=True)