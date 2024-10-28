import matplotlib.pyplot as plt
import numpy as np
figsize = (8, 3)

x_axis = np.array([1,3])
rule_labels = ["C2", "C3"]
# rule_labels = ["S2", "S3"]

Data = [[259.3, 254.0], [257.6, 247.3]]
model_labels = (["Type-C", "Conjunction"])
# Data = [[203.5, 200.9], [242.9, 248.9]]
# model_labels = (["Type-S", "Conjunction"])
colors = ["mediumorchid", "darkorange"]
spacing = np.array([-0.25, 0.25])

fig, ax = plt.subplots(1,1, figsize = figsize)

# Set the background color to grey
ax.set_facecolor('#EBEBEB')
# Draw white grid lines
ax.yaxis.grid(True, color='w', linestyle='solid')

for ind in range(len(x_axis)):
    ax.bar(x_axis + spacing[ind], Data[ind], label = model_labels[ind], color = colors[ind], width = 0.5, zorder = 3)
    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.legend(loc = "upper center")
ax.set_xticks(x_axis)
ax.set_xticklabels(rule_labels)
ax.set_ylim([150, 300])
ax.set_ylabel("Average BIC")
fig.savefig("Composite Results.png", format = "png", dpi = 500)    
# fig.savefig("Singular Results.png", format = "png", dpi = 500,)    
