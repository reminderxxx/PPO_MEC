# G14R20-F 未批准的下一步

本轮只关闭公共 startup handoff 的软件可操作性，不构成生产或科学批准。以下事项仍须由本实现 agent 之外的主体完成：

1. 独立 owner/verifier/issuer 确认安装 scope、预置 fixed-inode startup lock、authority key pin 与 continuity custody 运维流程。
2. 独立核验并决定是否采纳现时 release qualification 规则；历史独立 release attestation 仍为 unavailable。
3. 在不依赖执行器自报 hash 的条件下，custodian 核验新的 continuity checkpoint，并为下一宿主的新 challenge 签发匹配 receipt。
4. 另行签发 continuation approval；生产安装目前仍为 `None`，不得用 synthetic binding、CLI/env/proposal/approval 注入生产信任。
5. 只有上述外部资格与批准完成后，生产批准主体才可另行判断是否启动真实 continuation。G14R20-F 不授权恢复 v16、打开 holdout、执行 G14D/G15 或创建第二个正式 run。
