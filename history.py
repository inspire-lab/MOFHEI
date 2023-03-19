import sys
import matplotlib.pyplot as plt
import os

if len(sys.argv) > 1:
  filename = sys.argv[1]
else:
  files = [ f for f in os.listdir()  if f.startswith('shark_memlogger_') and not f.endswith('.png')]  
  timesstamps = [ f.replace('shark_memlogger_' ,'').replace('.log', '') for f in files ]
  timesstamps.sort(key=float)
  newest = timesstamps[-1]
  for f in files:
    if newest in f:
      filename = f
      break

with open(filename) as f:
  lines = f.readlines()
lines = [l.rstrip() for l in lines]

time = []
vms = []
rss={}
rss=[]
for l in lines:
  x = l.split(',')
  time.append(float(x[0].split(':')[1]))
  rss.append(float(x[1].split(':')[1]))
  vms.append(float(x[2].split(':')[1]))
rss_gb = [x/(1024*1024*1024)for x in rss]
vms_gb = [x/(1024*1024*1024)for x in vms]
plt.plot(time, vms_gb)
plt.plot(time, rss_gb)
plt.savefig(filename.replace('.log','.png'))

