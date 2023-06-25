import numpy as np

results = {'epoch_99.pth': {'TOP1': 53.527638190954775, 'TOP3': 72.80402010050251, 'TOP5': 78.05025125628141, 'TOP10': 85.78894472361809, 'TOP20': None, 'TOP50': None}, 'epoch_93.pth': {'TOP1': 53.527638190954775, 'TOP3': 72.48241206030151, 'TOP5': 78.35175879396985, 'TOP10': 85.7286432160804, 'TOP20': None, 'TOP50': None}, 'epoch_91.pth': {'TOP1': 53.894396115719196, 'TOP3': 72.85049565041473, 'TOP5': 78.85899251466721, 'TOP10': 86.04086587092858, 'TOP20': None, 'TOP50': None}, 'epoch_98.pth': {'TOP1': 53.4070351758794, 'TOP3': 72.78391959798995, 'TOP5': 78.25125628140704, 'TOP10': 85.60804020100502, 'TOP20': None, 'TOP50': None}, 'epoch_96.pth': {'TOP1': 53.165829145728644, 'TOP3': 72.38190954773869, 'TOP5': 78.25125628140704, 'TOP10': 85.96984924623115, 'TOP20': None, 'TOP50': None}, 'epoch_94.pth': {'TOP1': 53.246231155778894, 'TOP3': 72.38190954773869, 'TOP5': 78.45226130653266, 'TOP10': 86.19095477386935, 'TOP20': None, 'TOP50': None}, 'epoch_89.pth': {'TOP1': 53.06532663316583, 'TOP3': 73.26633165829146, 'TOP5': 79.15577889447236, 'TOP10': 86.87437185929649, 'TOP20': None, 'TOP50': None}, 'epoch_97.pth': {'TOP1': 53.246231155778894, 'TOP3': 72.72361809045226, 'TOP5': 78.15075376884423, 'TOP10': 85.68844221105527, 'TOP20': None, 'TOP50': None}, 'epoch_88.pth': {'TOP1': 52.50251256281407, 'TOP3': 72.66331658291458, 'TOP5': 78.8140703517588, 'TOP10': 86.35175879396985, 'TOP20': None, 'TOP50': None}, 'epoch_86.pth': {'TOP1': 52.94472361809045, 'TOP3': 73.22613065326634, 'TOP5': 78.91457286432161, 'TOP10': 86.57286432160804, 'TOP20': None, 'TOP50': None}, 'epoch_95.pth': {'TOP1': 53.185929648241206, 'TOP3': 72.58291457286433, 'TOP5': 78.2713567839196, 'TOP10': 85.98994974874371, 'TOP20': None, 'TOP50': None}, 'epoch_90.pth': {'TOP1': 53.30653266331658, 'TOP3': 73.1859296482412, 'TOP5': 79.01507537688443, 'TOP10': 86.41206030150754, 'TOP20': None, 'TOP50': None}, 'epoch_92.pth': {'TOP1': 53.20603015075377, 'TOP3': 72.7035175879397, 'TOP5': 78.4321608040201, 'TOP10': 86.09045226130654, 'TOP20': None, 'TOP50': None}, 'epoch_87.pth': {'TOP1': 52.904522613065325, 'TOP3': 73.60804020100502, 'TOP5': 79.31658291457286, 'TOP10': 86.67336683417085, 'TOP20': None, 'TOP50': None}}
















top1_acc = []
top3_acc = []
top5_acc = []
top10_acc = []

for epoch in range(90, 100):
    checkpoint_name = 'epoch_' + str(epoch) + '.pth'
    checkpoint = results[checkpoint_name]
    top1 = checkpoint['TOP1']
    top3 = checkpoint['TOP3']
    top5 = checkpoint['TOP5']
    top10 = checkpoint['TOP10']
    top1_acc.append(top1)
    top3_acc.append(top3)
    top5_acc.append(top5)
    top10_acc.append(top10)

print("{:.2f}".format(np.mean(top1_acc)), "{:.2f}".format(np.std(top1_acc)))
print("{:.2f}".format(np.mean(top3_acc)), "{:.2f}".format(np.std(top3_acc)))
print("{:.2f}".format(np.mean(top5_acc)), "{:.2f}".format(np.std(top5_acc)))
print("{:.2f}".format(np.mean(top10_acc)), "{:.2f}".format(np.std(top10_acc)))