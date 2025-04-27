# Validator

Validators download the models from hugging face for each miner based on the Bittensor chain metadata and continuously evaluate them, setting weights based on the performance of each model against a dataset for each competition. They also log results to [wandb](https://wandb.ai/macrocosmos/pretraining-validators).

You can view the entire validation system by reading the code in `neurons/validator.py`. Pseudocode for the validation system is as follows:

```python
    weights = zeros(256)
    while True:
        # Fetch random sample of batches to evaluate models on
        batches = get_random_sample_of_batches_from_dataset()
        
        # Fetch and or update models.
        models = get_and_update_models_from_miners()

        # Compute losses for each batch and each model
        model_losses = {}
        for model in models:
            for batch in batches:
                loss = get_loss_for_model_on_batch( model, batch )
                model_losses[ model ].append( loss )

        # Compute wins for models.
        model_wins = {}
        for model_a in models:
            for model_b in models:
                for i in len( batches )
                    # Determine if better model loss with relative block number boosting.
                    if iswin( model_losses[ model_a ][ i ], model_losses[ model_b ][ i ], block_a, block_b, epsilon = constants.timestamp_epsilon):
                        model_wins[ model_a ] += 1
                            
        # End epoch.
        # Weights are computed based on the ratio of wins a model attains during the epoch.
        for model_i in models:
            weights[ model_i ] += model_wins[ model_i ] / sum( model_wins.values() )
        weights = softmax( weights / temperature, dim=0 )

        # Set weights on the chain.
        set_weights( weight )
```

The behaviour of `iswin( loss_a, loss_b, block_a, block_b, epsilon_func, curr_block)` function intentionally skews the win function to reward models which have been hosted earlier such that newer models are only better than others iff their loss is `epsilon` percent lower accoring to the following function. `epsilon` is calculated based on a per-competition specified function based on the distance from the earlier model block to the current block.

```python
def iswin(loss_a, loss_b, block_a, block_b, epsilon_func, curr_block):
    loss_a = (1 - epsilon_func(curr_block, block_a)) * loss_a if block_a < block_b else loss_a
    loss_b = (1 - epsilon_func(curr_block, block_b)) * loss_b if block_b < block_a else loss_b
    return loss_a < loss_b
```

It is important to note that this affects the game theoretics of the incentive landscape since miners should only update their model (thus updating their timestamp to a newer date) if they have achieved an `epsilon` better loss on average on the Falcon Refined Web dataset than their previous model. This undermines the obvious optimal strategy for miners to copy the publicly available models from other miners. They **can** and should copy other miners, but they will always obtain fewer wins compared to them until they also decrease their loss by `epsilon`.

# System Requirements

Validators will need enough disk space to store the models of miners being evaluated. Each model has a max size by block defined in [constants/**init**.py](https://github.com/macrocosm-os/pretraining/blob/main/constants/__init__.py#L57) and the validator has cleanup logic to remove old models. It is recommended to have at least 2 TB of disk space and 80GB of system memory.

Validators will need enough processing power to evaluate their model. As of Sept 2nd, 2024, an upgrade to the Nvidia A100 GPU with 80GB of VRAM is required. This GPU's high throughput and FLOPs enable the running of 14B models without impacting the speed of the validation cycle. Although only 40GB of VRAM is necessary, we have observed that A100 GPUs with 80GB are more readily available and are offered at a comparable price to the 40GB variants. The additional VRAM provided by this GPU will allows more flexibility for optimization in future releases, enabling larger validation batch sizes to enhance the stability of the validation process by reducing scoring variance.

# Getting Started

## Prerequisites

1. Clone the repo

```shell
git clone https://github.com/macrocosm-os/pretraining.git
```

2. Setup your python [virtual environment](https://docs.python.org/3/library/venv.html) or [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

3. Install the requirements. From your virtual environment, run

```shell
cd pretraining
python -m pip install -e .
```

Note: flash-attn may not have their dependencies set up correctly. If you run into issues try installing those requirements separately first:

```shell
pip install packaging
pip install wheel
pip install torch
```

4. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).

5. (Optional) Run a Subtensor instance:

Your node will run better if you are connecting to a local Bittensor chain entrypoint node rather than using Opentensor's.
We recommend running a local node as follows and passing the ```--subtensor.network local``` flag to your running miners/validators.
To install and run a local subtensor node follow the commands below with Docker and Docker-Compose previously installed.

```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
docker compose up --detach
```

## Obtaining your Hugging Face and AWS access and secret tokens

The dataset for code, `The Stack V2-dedup`, requires a **Hugging Face access token** as well as **AWS access and secret keys**. Follow these steps to obtain and configure them:

**Step 1: Get Your Hugging Face Access Token**

1. Go to the [Hugging Face website](https://huggingface.co/).
2. If you don’t already have an account, create one. Otherwise, log in.
3. Go to the [dataset's website](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) and agree to their terms of use. You should immediately gain access to their dataset.
4. Click on your profile icon in the top-right corner, and select **Settings**.
5. In the settings menu, locate and click on **Access Tokens**.
6. Under the Access Tokens section, click **New token** and generate a token with write permissions.
7. Copy the generated token.

**Step 2: Obtaining your AWS Access and Secret keys**

1. Create a AWS account.
2. Follow these steps to retrieve your AWS root access keys

   1. **Access the AWS Management Console**  
      Navigate to the [AWS Management Console](https://aws.amazon.com/console/) and sign in using your root account credentials.

   2. **Navigate to Security Credentials**  
      Click on your account name located at the top right corner of the console. From the dropdown menu, select **"Security Credentials"**.

   3. **Proceed to Security Credentials**  
      A prompt may appear warning about accessing sensitive information. Click on **"Continue to Security Credentials"** to proceed.

   4. **Locate Access Keys**  
      In the **Security Credentials** section, expand the **"Access Keys (Access Key ID and Secret Access Key)"** tab to view your existing access keys.

   5. **Create a New Access Key**  
      To generate a new access key pair:
      - Click on **"Create New Access Key"**.
      - The new **Access Key ID** and **Secret Access Key** will be displayed.
      - View them on the screen or download them as a `.csv` file by clicking **"Download Key File"**.
      - Ensure you store these keys securely, as the **Secret Access Key** will not be retrievable after this point.
      - **Note:** If you do not save the **Secret Access Key** before closing the dialog, you will need to delete the access key and create a new one to retrieve the **Secret Access Key** again.
      - **Note:** AWS recommends avoiding the use of root access keys for everyday tasks. Instead, create IAM users with specific permissions to enhance security.

    For more information, visit [this guide](https://www.msp360.com/resources/blog/how-to-find-your-aws-access-key-id-and-secret-access-key/).

**Step 3: Create a `.env` file in the `pretraining` Directory**

1. Navigate to your `pretraining` directory where you want to save the environment file.
2. Create a new file named `.env` in this directory (if it doesn’t already exist). You can do this from the command line using:

   ```bash
   touch .env
   ```

3. Open the `.env` file with your preferred text editor and add the following lines:

    ```bash
    HF_TOKEN="<YOUR_HF_TOKEN_HERE>"
    AWS_ACCESS_KEY_ID="<YOUR_AWS_ACCESS_KEY_HERE>"
    AWS_SECRET_ACCESS_KEY="<YOUR_AWS_SECRET_KEY_HERE>"
    ```

4. Save and close the file.

This `.env` file now securely holds your access tokens, allowing scripts in the `pretraining` directory to load it automatically if they’re set up to read environment variables.

## Running the Validator

### With auto-updates

We highly recommend running the validator with auto-updates. This will help ensure your validator is always running the latest release, helping to maintain a high vtrust.

Prerequisites:

1. To run with auto-update, you will need to have [pm2](https://pm2.keymetrics.io/) installed.
2. Make sure your virtual environment is activated. This is important because the auto-updater will automatically update the package dependencies with pip.
3. Make sure you're using the main branch: `git checkout main`.

From the pretraining folder:

```shell
pm2 start --name net9-vali-updater --interpreter python scripts/start_validator.py -- --pm2_name net9-vali --wallet.name coldkey --wallet.hotkey hotkey [other vali flags]
```

This will start a process called `net9-vali-updater`. This process periodically checks for a new git commit on the current branch. When one is found, it performs a `pip install` for the latest packages, and restarts the validator process (who's name is given by the `--pm2_name` flag)

### Without auto-updates

If you'd prefer to manage your own validator updates...

From the pretraining folder:

```shell
pm2 start python -- ./neurons/validator.py --wallet.name coldkey --wallet.hotkey hotkey
```

# Configuration

## Flags

The Validator offers some flags to customize properties, such as the device to evaluate on and the number of models to evaluate each step.

You can view the full set of flags by running

```shell
python ./neurons/validator.py -h
```

## Test Running Validation

Test running validation:

```shell
python neurons/validator.py 
    --wallet.name YOUR_WALLET_NAME
    --wallet.hotkey YOUR_WALLET_HOTKEY 
    --device YOUR_CUDA DEVICE
    --wandb.off
    --offline
```

---
