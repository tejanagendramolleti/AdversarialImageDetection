print('Load modules...')
import foolbox
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import L2DeepFoolAttack, LinfBasicIterativeAttack, FGSM, L2CarliniWagnerAttack, PGD
import torch
from tqdm import tqdm
from collections import OrderedDict
from models.vgg_mnist import VGG as VGG_MNIST
from models.vgg_cif10 import VGG as VGG_CIF10
import argparse

# Processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--attack", default='fgsm', help="the attack method you want to use in order to create adversarial examples. Either fgsm, bim, pgd, df or cw")
parser.add_argument("--net", default='mnist', help="the dataset the net was trained on, either mnist, cif10, or cif100")
args = parser.parse_args()

# Choose attack
attack_method = args.attack
net = args.net

# Load model
print('Load model...')
if net == 'mnist':
    model = VGG_MNIST('VGG16')
    checkpoint = torch.load('./models/vgg_mnist.pth')
    if 'net' in checkpoint:
        checkpoint = checkpoint['net']
    model.load_state_dict(checkpoint)
    preprocessing = dict(mean=[0.1307], std=[0.3081], axis=-3)
elif net == 'cif10':
    model = VGG_CIF10('VGG16')
    checkpoint = torch.load('./models/vgg_cif10.pth')
    if 'net' in checkpoint:
        checkpoint = checkpoint['net']
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.` if it exists
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
else:
    print('Unknown net')
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = model.eval()

fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

# Load correctly classified data
testset = torch.load('./data/clean_data_' + net)

# Setup depending on attack
if attack_method == 'fgsm':
    attack = FGSM()
    epsilons = [0.03]
elif attack_method == 'bim':
    attack = LinfBasicIterativeAttack()
    epsilons = [0.03]
elif attack_method == 'pgd':
    attack = PGD()
    epsilons = [0.03]
elif attack_method == 'df':
    attack = L2DeepFoolAttack()
    epsilons = None
elif attack_method == 'cw':
    attack = L2CarliniWagnerAttack(steps=1000)
    epsilons = None
else:
    print('Unknown attack')
    exit()

images = []
images_advs = []
success_counter = 0

print('Perform attacks...')
for i in tqdm(range(len(testset))):
    image, label = testset[i]
    image = image.to(device)
    label = label.to(device)
    _, adv, success = attack(fmodel, image, criterion=foolbox.criteria.Misclassification(label), epsilons=epsilons)
    adv = adv[0]  # list to tensor
    success = success[0]
    if success:
        images_advs.append(adv.squeeze_(0).cpu())
        images.append(image.squeeze_(0).cpu())
        success_counter += 1

print('Attack success rate:', success_counter / len(testset))

torch.save(images, './data/' + net + '_adversarial_images/' + net + '_images_' + attack_method)
torch.save(images_advs, './data/' + net + '_adversarial_images/' + net + '_images_adv_' + attack_method)
print('Done performing attacks and adversarial examples are saved!')
