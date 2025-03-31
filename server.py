import flwr as fl

def start():
    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server(
        server_address="localhost:8081",  # 👈 Changed from 8080 to 8081
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy
    )

if __name__ == "__main__":
    print("✅ server.py started on port 8081")
    start()
