import matplotlib.pyplot as plt


# Before pruning
fig, ax = plt.subplots()
for p in points_int:
    plt.scatter(p[0], p[1], color="black")

for node in nodes.values():
    plt.scatter(node.position[0], node.position[1], color="blue")

for link in links.values():
    ni1, ni2 = link.node_indices
    p1 = nodes[str(ni1)].position
    p2 = nodes[str(ni2)].position
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="blue")
ax.set_aspect("equal")
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
t = np.linspace(-1, 1, 1000)
y = np.sqrt(1.0 - t**2)
ax.plot(t, y, color="black")
ax.plot(t, -y, color="black")


# Just network
fig, ax = plt.subplots()
for node in nodes.values():
    plt.scatter(node.position[0], node.position[1], color="blue")

for link in links.values():
    ni1, ni2 = link.node_indices
    p1 = nodes[str(ni1)].position
    p2 = nodes[str(ni2)].position
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="blue")
ax.set_aspect("equal")
# ax.set_xlim(-1.2, 1.2)
# ax.set_ylim(-1.2, 1.2)
# t = np.linspace(-1, 1, 1000)
# y = np.sqrt(1.0 - t**2)
# ax.plot(t, y, color="black")
# ax.plot(t, -y, color="black")
