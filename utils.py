
class DrawLine:
    def __init__(self, env="default", title="Title", xlabel="x", ylabel="y"):
        print(f"[DrawLine initialized] {title} | {xlabel} vs. {ylabel}")

    def __call__(self, xdata, ydata):
        print(f"[DrawLine] Episode {xdata}, Reward: {ydata:.2f}")
