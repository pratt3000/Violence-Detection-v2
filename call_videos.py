from eval_videos import modelDone

ob = modelDone()
ans = ob.evaluation("test/1.avi")

print("ans = ",ans)