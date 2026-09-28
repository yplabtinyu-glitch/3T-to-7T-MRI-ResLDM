from ldm.models.diffusion.resshift import LatentResShift
model = LatentResShift(resshift_T=15, resshift_p=0.3, resshift_kappa=2.0, **原本model.params)
model = model.to('cuda')
batch = next(iter(train_loader))
loss, loss_dict = model.shared_step_resshift(batch)
print('loss:', loss.item())  # 應該是有限數字,不是nan/inf,量級大概O(0.1~1)左右

x_start, y0, cond = model.get_input_resshift(batch, model.first_stage_key)
print('kappa=2.0 vs 實際residual std:', (y0 - x_start).std().item())  # 跟kappa差太多(>5x或<0.2x)要回頭調kappa
