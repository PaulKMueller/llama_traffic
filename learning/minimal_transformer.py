import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(0)
np.random.seed(0)

# This concept is also called teacher forceing.
# The flag decides if the loss will be calculted over all
# or just the predicted values.
calculate_loss_over_all_values = False

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = transformer_model(src, tgt)
#
# print(out)

input_window = 100
output_window = 5
batch_size = 10  # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = "Transformer"

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=10, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_sequences(input_data, time_window, output_window):
    """
    Creates input-output sequence pairs from time series data.

    Parameters:
    - input_data (numpy array): The time series data.
    - time_window (int): The number of time steps to be included in each input sequence.
    - output_window (int): The number of time steps to be predicted.

    Returns:
    - List of tuples, where each tuple contains an input sequence and its corresponding output sequence.
    """
    sequences = []
    length = len(input_data)
    for i in range(length - time_window):
        # Input sequence (the historical data window)
        input_sequence = np.append(
            input_data[i : i + time_window][:-output_window], output_window * [0]
        )

        # Output sequence (the target window)
        output_sequence = input_data[i : i + time_window]

        # Append the sequence pair to the list
        sequences.append((input_sequence, output_sequence))

    return torch.FloatTensor(sequences)


def prepare_data(input_window, output_window, device):
    """
    Prepares training and testing data for time series forecasting.

    Parameters:
    - input_window (int): The size of the input data window.
    - output_window (int): The size of the output data window.
    - device (torch.device): The device (CPU/GPU) where tensors will be sent.

    Returns:
    - Tuple of torch.FloatTensor, representing training and testing datasets.
    """
    # Generate synthetic time series data
    time = np.arange(0, 400, 0.1)
    amplitude = (
        np.sin(time)
        + np.sin(time * 0.05)
        + np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))
    )

    # Normalize the amplitude data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    amplitude_normalized = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)

    # Split the data into training and testing sets
    samples = 2800
    train_data = amplitude_normalized[:samples]
    test_data = amplitude_normalized[samples:]

    # Create input-output sequences for training and testing
    train_sequences = create_sequences(train_data, input_window, output_window)[
        :-output_window
    ]
    test_sequences = create_sequences(test_data, input_window, output_window)[
        :-output_window
    ]

    # Convert to PyTorch tensors and transfer to the specified device
    return train_sequences.to(device), test_sequences.to(device)


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i : i + seq_len]
    input = torch.stack(
        torch.stack([item[0] for item in data]).chunk(input_window, 1)
    )  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target


def train(train_data):
    model.train()  # Turn on the train mode
    total_loss = 0.0
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | "
                "lr {:02.6f} | {:5.2f} ms | "
                "loss {:5.5f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // batch_size,
                    scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()


def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.0
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            # look like the model returns static values for the output window
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(
                    output[-output_window:], target[-output_window:]
                ).item()

            test_result = torch.cat(
                (test_result, output[-1].view(-1).cpu()), 0
            )  # todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy()
    len(test_result)

    pyplot.plot(test_result, color="red")
    pyplot.plot(truth[:500], color="blue")
    pyplot.plot(test_result - truth, color="green")
    pyplot.grid(True, which="both")
    pyplot.axhline(y=0, color="k")
    # pyplot.savefig('graph/transformer-epoch%d.png'%epoch)
    pyplot.close()

    return total_loss / i


def predict_future(eval_model, data_source, steps):
    eval_model.eval()
    total_loss = 0.0
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    _, data = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps, 1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)

    pyplot.plot(data, color="red")
    pyplot.plot(data[:input_window], color="blue")
    pyplot.grid(True, which="both")
    pyplot.axhline(y=0, color="k")
    # pyplot.savefig('graph/transformer-future%d.png'%steps)
    pyplot.close()


# entweder ist hier ein fehler im loss oder in der train methode, aber die ergebnisse sind unterschiedlich
# auch zu denen der predict_future
def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.0
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            else:
                total_loss += (
                    len(data[0])
                    * criterion(output[-output_window:], targets[-output_window:])
                    .cpu()
                    .item()
                )
    return total_loss / len(data_source)


train_data, val_data = prepare_data()
model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.005
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

best_val_loss = float("inf")
epochs = 5  # The number of epochs
best_model = None

print(prepare_data()[0].shape)
print(prepare_data()[1].shape)

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)

    if epoch % 10 == 0:
        val_loss = plot_and_loss(model, val_data, epoch)
        predict_future(model, val_data, 200)
    else:
        val_loss = evaluate(model, val_data)

    print("-" * 89)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}".format(
            epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
        )
    )
    print("-" * 89)

    # if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    scheduler.step()

# src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number)
# out = model(src)
#
# print(out)
# print(out.shape)
