import akshare as ak

def resolve_akshare_symbol(keyword: str = "螺纹") -> str:
    """
    从 ak.futures_symbol_mark() 里按关键字找一个最匹配的 symbol
    keyword: 例如 '螺纹' '热卷' '铁矿' '豆粕'
    返回: 可直接用于 ak.futures_zh_realtime(symbol=...) 的 symbol
    """
    df = ak.futures_symbol_mark()

    # 尝试在可能的列里找关键字
    candidates = []
    for col in ["name", "名称", "品种", "商品", "symbol", "标记"]:
        if col in df.columns:
            sub = df[df[col].astype(str).str.contains(keyword, na=False)]
            if len(sub) > 0:
                candidates.append(sub)

    # 如果按名字列没找到，就退化到“全表字符串搜索”
    if not candidates:
        mask = df.astype(str).apply(lambda s: s.str.contains(keyword, na=False)).any(axis=1)
        sub = df[mask]
        if len(sub) == 0:
            raise ValueError(f"在 ak.futures_symbol_mark() 中找不到包含关键字 '{keyword}' 的行")
        candidates = [sub]

    sub = candidates[0]

    # 找到“symbol列”作为真正入参
    for sym_col in ["symbol", "标记", "market", "代码"]:
        if sym_col in sub.columns:
            return str(sub.iloc[0][sym_col])

    # 如果没有叫这些名字的列，就默认用第一列
    return str(sub.iloc[0, 0])
