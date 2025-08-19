import sys
sys.path.append('../')

from modular.Pretraining.LossMeasurment import calc_loss_batch,evaluate_model
from modular.GPTArchitecture.TextGeneration import generate_and_print_sample


def train_model_simple(model,train_loader,val_loader,optimizer,device,num_epochs,eval_freq,eval_iter,start_context,tokenizer):
    # initialize lists to track losses and tokens seen
    train_losses,val_losses,track_tokens_seen = [],[],[]
    tokens_seen,global_step = 0,-1
    
    # main training loop
    for epoch in range(num_epochs):
        model.train() # set model to training mode
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad() # reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward() #calculate loss gradients
            optimizer.step() # update model weights using loss gradients
            tokens_seen += input_batch.numel() #returns the total number of elements (or tokens) in the input_batch
            global_step += 1
            
            #optional evaluation step
            if global_step % eval_freq == 0:
                train_loss,val_loss = evaluate_model(
                    model,train_loader,val_loader,device,eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"EP {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        # Print a sample text after each epoch
        generate_and_print_sample(
            model,tokenizer,device,start_context
        )
    return train_losses,val_losses,track_tokens_seen
    