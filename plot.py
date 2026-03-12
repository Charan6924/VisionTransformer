import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('logs/metrics.csv')
print(df.head())

plt.plot(df['loss'])
plt.title('Train Loss')
plt.savefig('plots/trainloss.png')
plt.clf()

plt.plot(df['lr'])
plt.title('Learning rate over time')
plt.savefig('plots/lr.png')
plt.clf()

plt.plot(df['avg_epoch_loss'])
plt.title('Average Epoch Loss')
plt.savefig('plots/avg_epoch_loss.png')
plt.clf()

plt.plot(df['step'], df['elapsed_s'])
plt.xlabel('Step')
plt.ylabel('Elapsed Time (s)')
plt.title('Elapsed Time vs Step')
plt.savefig('plots/elapsed_s.png', dpi=150)
plt.clf()

plt.plot(df['step'], df['logit_scale'])
plt.xlabel('Step')
plt.ylabel('Logit Scale')
plt.title('Logit Scale vs Step')
plt.grid(True)
plt.ylim(df['logit_scale'].min() - 0.001, df['logit_scale'].max() + 0.001)
plt.savefig('plots/logit_scale.png')



