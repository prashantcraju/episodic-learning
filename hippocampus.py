import torch
import torch.nn as nn
import torch.optim as optim

class HippocampalHemisphere(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size):
        super(HippocampalHemisphere, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, memory_size),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(memory_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class LateralizedHippocampus(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size):
        super(LateralizedHippocampus, self).__init__()
        self.left_hemisphere = HippocampalHemisphere(input_size, hidden_size, memory_size)
        self.right_hemisphere = HippocampalHemisphere(input_size, hidden_size, memory_size)

    def forward(self, x):
        left_encoded, left_decoded = self.left_hemisphere(x)
        right_encoded, right_decoded = self.right_hemisphere(x)
        
        combined_encoded = (left_encoded + right_encoded) / 2
        combined_decoded = (left_decoded + right_decoded) / 2
        
        return combined_encoded, combined_decoded

    def decode(self, encoded):
        left_decoded = self.left_hemisphere.decoder(encoded)
        right_decoded = self.right_hemisphere.decoder(encoded)
        return (left_decoded + right_decoded) / 2

class EpisodicMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def add_episode(self, episode):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(episode)

    def retrieve_episodes(self):
        return self.memory

    def content_based_retrieval(self, query, k=1):
        if not self.memory:
            return None
        
        similarities = [torch.cosine_similarity(query, mem, dim=1).mean() for mem in self.memory]
        _, indices = torch.topk(torch.tensor(similarities), k)
        return [self.memory[i] for i in indices]


def train_model(model, episodes, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for episode in episodes:
            optimizer.zero_grad()
            encoded, decoded = model(episode)
            loss = criterion(decoded, episode)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(episodes)}")


def visualize_memories(encoded_episodes):
    # Flatten and concatenate the encoded episodes
    flattened_episodes = torch.cat([ep.view(1, -1) for ep in encoded_episodes], dim=0).numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embedded_episodes = tsne.fit_transform(flattened_episodes)
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    plt.scatter(embedded_episodes[:, 0], embedded_episodes[:, 1])
    plt.title("t-SNE visualization of encoded episodes")
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.show()

    
input_size = 100
hidden_size = 64
memory_size = 32
episode_length = 10
num_episodes = 100

model = LateralizedHippocampus(input_size, hidden_size, memory_size)
episodic_memory = EpisodicMemory(capacity=1000)

# Generate some random episodes
episodes = [torch.rand(episode_length, input_size) for _ in range(num_episodes)]

# Train the model
train_model(model, episodes, num_epochs=50, learning_rate=0.001)

# Store episodes in episodic memory
encoded_episodes = []
for episode in episodes:
    encoded, _ = model(episode)
    episodic_memory.add_episode(encoded.detach())
    encoded_episodes.append(encoded.detach())

# Visualize encoded memories
visualize_memories(encoded_episodes)

# Content-based retrieval (as before)
query = torch.rand(episode_length, input_size)
encoded_query, _ = model(query)
retrieved_episodes = episodic_memory.content_based_retrieval(encoded_query, k=3)

if retrieved_episodes:
    print("Top 3 similar episodes retrieved:")
    reconstructed_episodes = []
    for i, episode in enumerate(retrieved_episodes):
        decoded = model.decode(episode)
        error = nn.MSELoss()(decoded, episodes[i]).item()
        print(f"Episode {i+1} reconstruction error: {error:.6f}")
        reconstructed_episodes.append(decoded)
    
    # Analyze query reconstruction
    query_error = analyze_query_reconstruction(model, query)
    print(f"Query reconstruction error: {query_error:.6f}")
    
    # Visualize original vs reconstructed episodes
    visualize_reconstructions(episodes[:3], reconstructed_episodes)
else:
    print("No episodes retrieved.")