import pandas as pd
import torch
from constants import *


class BiasedMatrixFactorization(torch.nn.Module):
  def __init__(self, n_users, n_items, n_factors=1, use_global_intercept=True):
    super().__init__()
    self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=False)
    self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=False)
    self.user_intercepts = torch.nn.Embedding(n_users, 1, sparse=False)
    self.item_intercepts = torch.nn.Embedding(n_items, 1, sparse=False)
    self.use_global_intercept = use_global_intercept
    self.global_intercept = torch.nn.parameter.Parameter(torch.zeros(1, 1))
    torch.nn.init.xavier_uniform_(self.user_factors.weight)
    torch.nn.init.xavier_uniform_(self.item_factors.weight)
    self.user_intercepts.weight.data.fill_(0.0)
    self.item_intercepts.weight.data.fill_(0.0)

  def forward(self, user, item):
    pred = self.user_intercepts(user) + self.item_intercepts(item)
    pred += (self.user_factors(user) * self.item_factors(item)).sum(1, keepdim=True)
    if self.use_global_intercept == True:
      pred += self.global_intercept
    return pred.squeeze()


def run_mf(
  ratings,
  l2_lambda,
  l2_intercept_multiplier,
  numFactors,
  epochs,
  useGlobalIntercept,
  runName="prod",
  logging=True,
):
  assert (numFactors == 1)
  noteData = ratings
  noteData.dropna(0, inplace=True)

  noteIdMap = (
    pd.DataFrame(noteData[noteIdKey].unique())
    .reset_index()
    .set_index(0)
    .reset_index()
    .rename(columns={0: noteIdKey, "index": noteIndexKey})
  )
  raterIdMap = (
    pd.DataFrame(noteData[raterParticipantIdKey].unique())
    .reset_index()
    .set_index(0)
    .reset_index()
    .rename(columns={0: raterParticipantIdKey, "index": raterIndexKey})
  )

  noteRatingIds = noteData.merge(noteIdMap, on=noteIdKey)
  noteRatingIds = noteRatingIds.merge(raterIdMap, on=raterParticipantIdKey)

  n_users = noteRatingIds[raterIndexKey].nunique()
  n_items = noteRatingIds[noteIndexKey].nunique()
  if logging:
    print("------------------")
    print(f"Users: {n_users}, Notes: {n_items}")

  criterion = torch.nn.MSELoss()

  l2_lambda_intercept = l2_lambda * l2_intercept_multiplier

  rating = torch.FloatTensor(noteRatingIds[helpfulNumKey].values)
  row = torch.LongTensor(noteRatingIds[raterIndexKey].values)
  col = torch.LongTensor(noteRatingIds[noteIndexKey].values)

  mf_model = BiasedMatrixFactorization(
    n_users, n_items, use_global_intercept=useGlobalIntercept, n_factors=numFactors
  )
  optimizer = torch.optim.Adam(mf_model.parameters(), lr=1)  # learning rate

  def print_loss():
      y_pred = mf_model(row, col)
      train_loss = criterion(y_pred, rating)

      if logging:
        print("epoch", epoch, loss.item())
        print("TRAIN FIT LOSS: ", train_loss.item())

  for epoch in range(epochs):
    # Set gradients to zero
    optimizer.zero_grad()

    # Predict and calculate loss
    y_pred = mf_model(row, col)
    loss = criterion(y_pred, rating)
    l2_reg_loss = torch.tensor(0.0)

    for name, param in mf_model.named_parameters():
      if "intercept" in name:
        l2_reg_loss += l2_lambda_intercept * (param ** 2).mean()
      else:
        l2_reg_loss += l2_lambda * (param ** 2).mean()

    loss += l2_reg_loss

    # Backpropagate
    loss.backward()

    # Update the parameters
    optimizer.step()

    if epoch % 50 == 0:
      print_loss()

  print_loss()

  assert mf_model.item_factors.weight.data.numpy().shape[0] == noteIdMap.shape[0]

  noteIdMap[noteFactor1Key] = mf_model.item_factors.weight.data.numpy()[:, 0]
  raterIdMap[raterFactor1Key] = mf_model.user_factors.weight.data.numpy()[:, 0]
  noteIdMap[noteInterceptKey] = mf_model.item_intercepts.weight.data.numpy()
  raterIdMap[raterInterceptKey] = mf_model.user_intercepts.weight.data.numpy()

  globalIntercept = None
  if useGlobalIntercept:
    globalIntercept = mf_model.global_intercept

  return noteIdMap, raterIdMap, globalIntercept
