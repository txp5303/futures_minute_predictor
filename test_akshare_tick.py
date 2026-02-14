from core.market_data_akshare import AKShareMarketDataSource

if __name__ == "__main__":
    mds = AKShareMarketDataSource()

    # 先用一个你想看的品种（这里给你一个占位写法）
    # 如果报错/没数据，我们就根据返回 df 的列与可用合约来调整 symbol
    symbol = "RB0"  # 或 "RB9999" / "RB主连" 等（不同源写法不同）
    tick = mds.get_last_tick(symbol)
    print(tick)
