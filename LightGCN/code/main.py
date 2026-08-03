if __name__ == '__main__':
    import os
    import world
    import utils
    import torch
    import time
    import datetime
    import Procedure
    import register
    import wandb
    from utils import Timer
    from os.path import join
    from dotenv import load_dotenv
    from register import dataset

    utils.set_seed(world.config['seed'])
    if not world.config['resume']:
        print(">> SEED:", world.config['seed'])

    RecModel = register.MODELS[world.config['model_name']](world.config, dataset)
    RecModel = RecModel.to(world.config['device'])
    bpr = utils.BPRLoss(RecModel, world.config)

    weight_file = utils.getFileName()
    start_epoch = 0
    wandb_id = None
    print(f"Load and save to [{weight_file}]")
    if world.config['pretrain'] or world.config['resume']:
        try:
            checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
            RecModel.load_state_dict(checkpoint['state_dict'])
            if world.config['resume']:
                start_epoch = checkpoint['epoch'] + 1
                wandb_id = checkpoint["wandb_id"]
            print(f"Loaded model weights from [{weight_file}]")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning.")

    # Weight and bias initialization
    if world.config['wandb']:
        load_dotenv()
        os.environ["WANDB_SILENT"] = "true"
        wandb_config = world.config.copy()
        wandb_config['device'] = str(wandb_config['device'])
        run = wandb.init(
            entity="mati_gar-university-of-wroclaw",
            project="Engineer Thesis",
            name=utils.getName(),
            group=f"{world.config['experiment']}-new",
            config=wandb_config,
            id=wandb.util.generate_id() if wandb_id is None else wandb_id,
            resume="allow"
        )
        run.watch(RecModel, log="all")
        run.define_metric("metrics/*", summary="max")    

    start_time = time.time() # Track START time
    if world.config['resume']:
        print("### LightGCN training resumed at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    else:
        print("### LightGCN training started at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    
    if world.config['pin_sampling'] > 0:
        with Timer(name="Graph construction"):
            dgl_graph = dataset.get_dgl_graph()
        time_info = Timer.dict(["Graph construction"])
        Timer.zero(["Graph construction"])
        print(time_info)
    
    early_stopping = 0

    try:
        if world.config['resume']:
            best_results = checkpoint['best_results']
        else:
            best_results = {
                f"{metric}@{k}": {'best_value': 0, 'best_epoch': 0}
                for metric in world.all_metrics
                for k in world.config['topKs']
            }
            best_results["loss"] = {'best_value': float('inf'), 'best_epoch': 0}

        for epoch in range(start_epoch, world.config['epochs']):
            start = time.time()
            
            if world.config['pin_sampling'] > 0 and epoch % world.config['pin_sampling'] == 0:
                with Timer(name="PinSage sampling"):
                    pinsage_graph = dataset.get_pin_graph(
                        dgl_graph,
                        n_traces=world.config['pin_n_traces'], 
                        n_hops=world.config['pin_n_hops'], 
                        top_k=world.config['pin_top_k']
                    )
                    RecModel.Graph = pinsage_graph
                time_info = Timer.dict(["PinSage sampling"])
                Timer.zero(["PinSage sampling"])
                print(time_info)

            loss, time_info, bpr_sampled_nodes = Procedure.BPR_train_original(dataset, RecModel, bpr, epoch, neg_k=world.config['neg_k'])
            print(f"EPOCH[{epoch + 1}/{world.config['epochs']}]: Loss: {loss} {time_info}")
            
            with Timer(name="Testing"):
                results = Procedure.Test(dataset, RecModel, epoch, world.config['multicore'])
            
            time_info = Timer.dict(["Testing"])
            Timer.zero(["Testing"])
            print(f"{time_info}:")
            wandb_results, best_results, is_best = utils.print_and_update_metrics(results, best_results, loss, epoch)
            
            if is_best:
                early_stopping = 0
                files = [weight_file, utils.getFileName(is_best=True)]
            else:
                if world.config['early_stopping'] > 0:
                    early_stopping += 1
                    print(f"Early stopping: {early_stopping}/{world.config['early_stopping']}")
                files = [weight_file]
            for file in files:
                torch.save({
                        "state_dict": RecModel.state_dict(),
                        "epoch": epoch,
                        "wandb_id": run.id if world.config['wandb'] else None,
                        "best_results": best_results,
                        "training_duration": time.time() - start_time
                    },
                    file
                )
            
            l2_norm, cos_drift = utils.print_and_aggravate_node_dist(RecModel, dataset.item_decile_groups, "Item")
            utils.print_and_aggravate_node_dist(RecModel, dataset.user_decile_groups, "User")
            bpr_nodes_groups = utils.print_and_aggravate_node_occurance(bpr_sampled_nodes, dataset.item_to_decile_group, location="BPR")
            
            model_nodes_groups = {}
            if world.config['model_name'] == 'flgcn':
                model_nodes_groups = utils.print_and_aggravate_node_occurance(RecModel.sampled_nodes, dataset.item_to_decile_group, location="Model")
                RecModel.sampled_nodes.clear()
            
            if world.config['wandb']:
                run.log({
                    "epoch": epoch,
                    "loss": loss,
                    **l2_norm,
                    **cos_drift,
                    **bpr_nodes_groups,
                    **model_nodes_groups,
                    **wandb_results
                })
            
            if world.config['early_stopping'] > 0:
                if early_stopping == world.config['early_stopping']:
                    break
    finally:
        print("### Best results:")
        for metric_str, result_data in best_results.items():
            best_value = result_data['best_value']
            best_epoch = result_data['best_epoch']
            print(f"- {metric_str}: {best_value} - epoch: {best_epoch}")

        if world.config['wandb']:
            run.summary["best_epochs"] = {metric_str: result_data['best_epoch'] for metric_str, result_data in best_results.items()}

        best_checkpoint = torch.load(utils.getFileName(is_best=True), map_location=torch.device('cpu'))
        RecModel.load_state_dict(best_checkpoint['state_dict'])
        with Timer(name="Group testing"):
            group_results, groups_n_users = Procedure.Test(
                dataset, RecModel, epoch, world.config['multicore'], 
                groups=dataset.item_decile_groups
            )
        time_info = Timer.dict(["Group testing"])
        Timer.zero(["Group testing"])
        
        print(f"### Group results: {time_info}")
        group_results = utils.print_and_parse_group_metrics(group_results, groups_n_users)
        
        if world.config['wandb']:
            run.summary["group_results"] = group_results

        # Track END time
        end_time = time.time()
        duration = end_time - start_time
        
        if world.config['resume']:
            duration += checkpoint['training_duration']
            
        print("### LightGCN training ended at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
        print("### Total training time:", str(datetime.timedelta(seconds=duration)).split(".")[0])

        if world.config['wandb']:
            run.finish()