import torch
import argparse
import break22
from tool import utils
from model import model_text, model_t_y_dis, model_t_y_dis_rep_3, model_t_y_dis_rep_3_rerank, model_t_y_dis_rep_3_weight

parser = argparse.ArgumentParser()
args = utils.get_parsere(parser)

#2
# break22.train('save/notop_noipc/2/text/600_0.13638675049737548_0.13638675049737548.pth', args = args, address='za_2_text/', net=model_text.net(args).to(args.device))
# break22.train('save/notopk_subipc/2/dis_2/500_0.2231996876610951_0.2231996876610951.pth', args = args, address='za_2_dis/', net=model_t_y_dis.net(args).to(args.device))
# break22.train('save/notop_noipc/2/dis/700_0.10527952203287207_0.10527952203287207.pth', args = args, address='za_2_dis/', net=model_t_y_dis_rep_3.net(args).to(args.device))
# break22.train('', args = args, address='za_2_rerank/', net=model_t_y_dis_rep_3_rerank.net(args).to(args.device))
# break22.train('save/notop_noipc/2/weight/1400_0.10545144111313151_0.10545144111313151.pth', args = args, address='za_2_weight/', net=model_t_y_dis_rep_3_weight.net(args).to(args.device))

#62
break22.train('save/topk_subipc/62/text/200_0.09085325585105099_0.09085325585105099.pth', args = args, address='za_62_text/', net=model_text.net(args).to(args.device))
# break22.train('save/notopk_subipc/12/dis_2/500_0.2375217026833332_0.2375217026833332.pth', args = args, address='za_12_dis/', net=model_t_y_dis.net(args).to(args.device))
break22.train('save/topk_subipc/62/dis/400_0.0742593491420102_0.0742593491420102.pth', args = args, address='za_62_dis/', net=model_t_y_dis_rep_3.net(args).to(args.device))
# break22.train('', args = args, address='za_12_rerank/', net=model_t_y_dis_rep_3_rerank.net(args).to(args.device))
break22.train('save/topk_subipc/62/weight/300_0.07323481630569753_0.07323481630569753.pth', args = args, address='za_62_weight/', net=model_t_y_dis_rep_3_weight.net(args).to(args.device))

#22
# break22.train('save/notop_noipc/22/text/400_0.14979331112805416_0.14979331112805416.pth', args = args, address='za_22_text/', net=model_text.net(args).to(args.device))
# break22.train('save/notopk_subipc/22/dis_2/600_0.2303996067211935_0.2303996067211935.pth', args = args, address='za_22_dis/', net=model_t_y_dis.net(args).to(args.device))
# break22.train('save/notop_noipc/22/dis/800_0.09688552191664436_0.09688552191664436.pth', args = args, address='za_22_dis/', net=model_t_y_dis_rep_3.net(args).to(args.device))
# break22.train('', args = args, address='za_22_rerank/', net=model_t_y_dis_rep_3_rerank.net(args).to(args.device))
# break22.train('save/notop_noipc/22/weight/1300_0.09369771260923396_0.09369771260923396.pth', args = args, address='za_22_weight/', net=model_t_y_dis_rep_3_weight.net(args).to(args.device))

#82
# break22.train('save/topk_subipc/82/text/300_0.08845910215630937_0.08845910215630937.pth', args = args, address='za_82_text/', net=model_text.net(args).to(args.device))
# break22.train('save/notopk_subipc/2/dis_2/500_0.2231996876610951_0.2231996876610951.pth', args = args, address='za_2_dis/', net=model_t_y_dis.net(args).to(args.device))
# break22.train('save/topk_subipc/82/dis/250_0.08289956198169444_0.08289956198169444.pth', args = args, address='za_82_dis/', net=model_t_y_dis_rep_3.net(args).to(args.device))
# break22.train('save/700_0.10792836997178241_0.10792836997178241.pth', args = args, address='za_32_rerank/', net=model_t_y_dis_rep_3_rerank.net(args).to(args.device))
# break22.train('save/topk_subipc/82/weight/250_0.0781891467467619_0.0781891467467619.pth', args = args, address='za_82_weight/', net=model_t_y_dis_rep_3_weight.net(args).to(args.device))

#42
# break22.train('save/notop_noipc/42/text/500_0.14336712302244975_0.14336712302244975.pth', args = args, address='za_42_text/', net=model_text.net(args).to(args.device))
# break22.train('save/notopk_subipc/12/dis_2/500_0.2375217026833332_0.2375217026833332.pth', args = args, address='za_12_dis/', net=model_t_y_dis.net(args).to(args.device))
# break22.train('save/notop_noipc/42/dis/1100_0.10537123938556761_0.10537123938556761.pth', args = args, address='za_42_dis/', net=model_t_y_dis_rep_3.net(args).to(args.device))
# break22.train('save/notop_noipc/42/rerank/1000_0.11121848028604732_0.11121848028604732.pth', args = args, address='za_42_rerank/', net=model_t_y_dis_rep_3_rerank.net(args).to(args.device))
# break22.train('save/notop_noipc/42/weight/1900_0.10293977351887434_0.10293977351887434.pth', args = args, address='za_42_weight/', net=model_t_y_dis_rep_3_weight.net(args).to(args.device))

