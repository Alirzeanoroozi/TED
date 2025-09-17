from model import build_transformer_esmc
from dataset import causal_mask
from config import get_config, get_weights_file_path
from dataloader import get_ds

import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import wandb

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
device = torch.device(device)

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type(torch.int64).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).fill_(next_word.item()).type(torch.int64).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_tgt, max_len, device, print_msg, num_examples=10):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in validation_ds:
            count += 1

            model_out = greedy_decode(model, batch["src_text"], batch["encoder_mask"].to(device), tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = ""
            for i in model_out.detach().tolist():
                model_out_text += tokenizer_tgt.decode([i])

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*80)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*80)
                break
    
    # Calculate character-wise accuracy for each prediction
    def charwise_acc(pred, exp):
        if not exp:
            return 0.0
        # Pad the shorter string so zip works for all chars
        max_len = max(len(pred), len(exp))
        pred = pred.ljust(max_len)
        exp = exp.ljust(max_len)
        correct = sum(p == e for p, e in zip(pred, exp))
        return correct / max_len

    charwise_accs = [charwise_acc(p, e) for p, e in zip(predicted, expected)]

    table = wandb.Table(columns=["input", "label", "predicted", "charwise_accuracy"])
    for inp, lab, pr, acc in zip(source_texts, expected, predicted, charwise_accs):
        table.add_data(inp, lab, pr, acc)
    wandb.log({"eval_samples": table})

def train_model(config):
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = build_transformer_esmc(tokenizer_tgt.get_vocab_size(), config['tgt_seq_len'], d_model=config['d_model']).to(device)
    
    # Print the number of trainable parameters in the model and all the numbers
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    global_step = 0
    def lr_lambda(step):
        return max(0.1 ** (step // 5000), 1e-3)
    # scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=global_step-1)

    # If the user specified a model to preload before training, load it
    model_filename = get_weights_file_path(config) if config['preload'] else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        # optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        # scheduler.load_state_dict(state['scheduler_state_dict'])
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]')).to(device)

    batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {global_step:02d}")
    for batch in batch_iterator:
        torch.cuda.empty_cache()
        model.train()
        encoder_input = batch['encoder_input'].to(device) # (B, src_seq_len)
        decoder_input = batch['decoder_input'].to(device) # (B, tgt_seq_len)
        encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, src_seq_len)
        decoder_mask = batch['decoder_mask'].to(device) # (B, 1, tgt_seq_len, tgt_seq_len)

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = model.encode(batch['src_text'], encoder_mask) # (B, src_seq_len, d_model)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, tgt_seq_len, d_model)
        proj_output = model.project(decoder_output) # (B, tgt_seq_len, vocab_size)

        # Compare the output with the label
        label = batch['label'].to(device) # (B, tgt_seq_len)

        # Compute the loss using a simple cross entropy
        loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        # Log the loss
        wandb.log({'train loss': loss.item(), 'global_step': global_step})

        # Backpropagate the loss
        loss.backward()

        # Update the weights
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # scheduler.step()

        global_step += 1
        if global_step % config['val_epochs'] == 0:
            run_validation(model, val_dataloader, tokenizer_tgt, 100, device, lambda msg: batch_iterator.write(msg))

            # Save the model at the end of every epoch
            model_filename = get_weights_file_path(config)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                # 'scheduler_state_dict': scheduler.state_dict()
            }, model_filename.split('.')[0] + "_new.pt")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    # Make sure the weights folder exists
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    wandb.init(project="transformer_scratch", config=config)
    train_model(config)
