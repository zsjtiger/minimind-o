"""按配比混合多个数据源。

目标数量 = total × weight / Σweight。数据源不足时有放回地重复采样（过采样），并在报告中标出。
"""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def mix_sources(sources, total, out_path, seed=0, report_path=None):
    """sources: [{"name", "path", "weight"}]。返回每个数据源的统计报告。"""
    wsum = sum(s['weight'] for s in sources)
    if wsum <= 0:
        raise ValueError('权重之和必须大于 0')
    rng = np.random.default_rng(seed)
    tables, report = [], {}
    for s in sources:
        n_avail = pq.ParquetFile(s['path']).metadata.num_rows
        target = int(round(total * s['weight'] / wsum))
        entry = {'weight': s['weight'], 'available': n_avail, 'target': target, 'upsampled': False}
        report[s['name']] = entry
        if n_avail == 0 or target == 0:
            entry['taken'] = 0
            continue
        replace = target > n_avail
        entry['upsampled'] = replace
        idx = np.sort(rng.choice(n_avail, size=target, replace=replace))
        tables.append(pq.read_table(s['path']).take(pa.array(idx)))
        entry['taken'] = target
    if not tables:
        raise ValueError('没有可用的数据')
    combined = pa.concat_tables(tables, promote_options='default')
    combined = combined.take(pa.array(rng.permutation(combined.num_rows)))
    pq.write_table(combined, out_path)
    total_taken = combined.num_rows
    for entry in report.values():
        entry['ratio'] = round(entry['taken'] / total_taken, 4) if total_taken else 0.0
    if report_path:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({'total': total_taken, 'sources': report}, f, ensure_ascii=False, indent=2)
    return report
