import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

target_name = "gpt2-medium"
draft_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(target_name)
target_model = AutoModelForCausalLM.from_pretrained(target_name).to(device)
draft_model = AutoModelForCausalLM.from_pretrained(draft_name).to(device)

prompt = input("Enter your prompt: ")
num_tokens_wanted = int(input("Enter the number of tokens you want to generate: "))

num_tokens_generated = 0
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
generated = input_ids

total_accepted = 0
total_rounds = 0
start_time = time.time()

while num_tokens_generated < num_tokens_wanted:
  n = generated.shape[1]

  # Draft Generation
  past_key_values = None
  current_token = generated
  num_new_tokens = 5 #k=5
  candidate_tokens = []
  candidate_probs = []

  for step in range(num_new_tokens):
    with torch.no_grad():
      outputs = draft_model(current_token, past_key_values=past_key_values, use_cache=True)
      next_token_logits = outputs.logits[0, -1, :]

      probs = torch.softmax(next_token_logits, dim=-1)
      candidate_probs.append(probs)

      next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
      candidate_tokens.append(next_token_id)
      past_key_values = outputs.past_key_values
      current_token = next_token_id

  # Target Verification
  verify_input = torch.cat([generated] + candidate_tokens, dim=1)
  outputs = target_model(verify_input)
  token_logits = outputs.logits[0, n-1:n-1+num_new_tokens, :]
  probs = torch.softmax(token_logits, dim=-1)
  num_accepted = 0
  rejected = False

  for i in range(len(candidate_tokens)):
    token_id = candidate_tokens[i].item()
    q_x = candidate_probs[i][token_id]
    p_x = probs[i][token_id]

    accept_prob = min(1.0, p_x/q_x)
    if torch.rand(1).item() < accept_prob:
      num_accepted += 1
      continue
    else:
      adjusted = torch.clamp(probs[i] - candidate_probs[i], 0)
      adjusted = adjusted/adjusted.sum()
      resampled_token_id = torch.multinomial(adjusted, num_samples=1).unsqueeze(0)
      rejected = True
      break

  if rejected:
    generated = torch.cat([generated] + candidate_tokens[:num_accepted] + [resampled_token_id], dim=1)
  else:
    generated = torch.cat([generated] + candidate_tokens, dim=1)
    bonus_token_logits = outputs.logits[0, n-1+num_new_tokens, :]
    bonus_token_id = torch.argmax(bonus_token_logits).unsqueeze(0).unsqueeze(0)
    generated = torch.cat([generated, bonus_token_id], dim=1)

  num_tokens_generated = generated.shape[1] - input_ids.shape[1]
  total_accepted += num_accepted
  total_rounds += 1

elapsed = time.time() - start_time

print(tokenizer.decode(generated[0]))
print(f"Tokens generated: {num_tokens_generated}")
print(f"Rounds: {total_rounds}")
print(f"Avg accepted per round: {total_accepted / total_rounds:.2f} / {num_new_tokens}")
print(f"Time: {elapsed:.2f}s")
print(f"Tokens/sec: {num_tokens_generated / elapsed:.2f}")
