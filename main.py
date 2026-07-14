import os
import re
import glob

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="연령별 인구 구조 대시보드",
    page_icon="👥",
    layout="wide",
)

st.title("👥 연령별 인구 구조 대시보드")
st.caption("행정안전부 · 연령별 인구현황 데이터를 기반으로 지역별 인구 구조를 살펴봅니다.")


# =========================================================
# 데이터 로드 & 전처리
# =========================================================
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    if not csv_files:
        st.error("main.py와 같은 폴더에서 CSV 파일을 찾을 수 없습니다. 데이터 파일을 함께 업로드해주세요.")
        st.stop()
    path = csv_files[0]

    df = pd.read_csv(path, encoding="cp949", thousands=",", low_memory=False)

    # ---- 지역명 / 행정코드 분리 ----
    def parse_region(raw):
        s = str(raw).strip()
        m = re.match(r"^(.*?)\s*\((\d+)\)\s*$", s)
        if m:
            return m.group(1).strip(), m.group(2)
        return s, None

    parsed = df.iloc[:, 0].apply(parse_region)
    df["지역명"] = parsed.apply(lambda x: x[0])
    df["행정코드"] = parsed.apply(lambda x: x[1])

    # ---- 행정구역 단위(레벨) 판별 ----
    def get_level(code):
        if code is None:
            return "기타"
        if code.endswith("00000000"):
            return "시도"
        if code.endswith("000000"):
            return "시군구"
        return "읍면동"

    df["레벨"] = df["행정코드"].apply(get_level)

    # ---- 연령별 컬럼 동적 추출 (계 / 남 / 여) ----
    def extract_age_cols(gender_tag):
        cols = {}
        pat_age = re.compile(rf"_{gender_tag}_(\d+)세$")
        pat_100 = re.compile(rf"_{gender_tag}_100세 이상$")
        for c in df.columns:
            m = pat_age.search(c)
            if m:
                cols[int(m.group(1))] = c
            elif pat_100.search(c):
                cols[100] = c
        return dict(sorted(cols.items()))

    total_cols = extract_age_cols("계")
    male_cols = extract_age_cols("남")
    female_cols = extract_age_cols("여")

    ages = sorted(total_cols.keys())

    total_pop_col = [c for c in df.columns if c.endswith("_계_총인구수")][0]
    df["총인구수"] = df[total_pop_col]

    total_matrix = df[[total_cols[a] for a in ages]].to_numpy(dtype=float)
    male_matrix = df[[male_cols[a] for a in ages]].to_numpy(dtype=float)
    female_matrix = df[[female_cols[a] for a in ages]].to_numpy(dtype=float)

    meta = df[["지역명", "행정코드", "레벨", "총인구수"]].reset_index(drop=True)

    # 연령별 비율(정규화) 매트릭스 - 인구 구조 비교/유사도 계산용
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = total_matrix.sum(axis=1, keepdims=True)
        pct_matrix = np.divide(
            total_matrix, row_sums,
            out=np.zeros_like(total_matrix),
            where=row_sums != 0,
        )

    return meta, ages, total_matrix, male_matrix, female_matrix, pct_matrix


meta, ages, total_matrix, male_matrix, female_matrix, pct_matrix = load_data()


def region_picker(meta, key_prefix, default_level="시군구"):
    """사이드바/탭 공용 지역 선택 위젯. 선택된 행의 인덱스를 반환한다."""
    levels = ["시도", "시군구", "읍면동"]
    level = st.selectbox(
        "행정구역 단위", levels,
        index=levels.index(default_level),
        key=f"{key_prefix}_level",
    )
    filtered = meta[meta["레벨"] == level]

    search = st.text_input("지역명 검색 (예: 강남, 해운대, 청주)", key=f"{key_prefix}_search")
    if search:
        filtered = filtered[filtered["지역명"].str.contains(search, case=False, na=False)]

    if filtered.empty:
        st.warning("검색 결과가 없습니다.")
        return None, level

    options = filtered.index.tolist()
    idx = st.selectbox(
        "지역 선택",
        options=options,
        format_func=lambda i: f"{meta.loc[i, '지역명']}  ·  총인구 {int(meta.loc[i, '총인구수']):,}명",
        key=f"{key_prefix}_region",
    )
    return idx, level


# =========================================================
# 탭 구성
# =========================================================
tab1, tab2, tab3 = st.tabs(["👤 인구 피라미드", "📊 연령별 구조 비교", "👯 쌍둥이 지역 찾기"])

# ---------------------------------------------------------
# TAB 1. 인구 피라미드
# ---------------------------------------------------------
with tab1:
    st.subheader("성별·연령별 인구 피라미드")
    col_side, col_main = st.columns([1, 3])

    with col_side:
        idx, _ = region_picker(meta, key_prefix="t1")

    with col_main:
        if idx is not None:
            region_name = meta.loc[idx, "지역명"]
            m_vals = male_matrix[idx]
            f_vals = female_matrix[idx]
            max_val = max(m_vals.max(), f_vals.max(), 1)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=ages, x=-m_vals, orientation="h", name="남성",
                marker_color="#4C72B0",
                customdata=m_vals,
                hovertemplate="%{y}세 · 남성 %{customdata:,.0f}명<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                y=ages, x=f_vals, orientation="h", name="여성",
                marker_color="#DD8452",
                customdata=f_vals,
                hovertemplate="%{y}세 · 여성 %{customdata:,.0f}명<extra></extra>",
            ))

            tick_step = max(int(max_val // 5), 1)
            tickvals = list(range(-int(max_val), int(max_val) + tick_step, tick_step))
            ticktext = [f"{abs(v):,}" for v in tickvals]

            fig.update_layout(
                title=f"{region_name} 인구 피라미드",
                barmode="overlay",
                bargap=0.05,
                template="plotly_white",
                height=750,
                hovermode="y unified",
                xaxis=dict(title="인구수(명)", tickvals=tickvals, ticktext=ticktext,
                           range=[-max_val * 1.1, max_val * 1.1]),
                yaxis=dict(title="연령", dtick=5),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
            )
            st.plotly_chart(fig, use_container_width=True)

            total_pop = int(meta.loc[idx, "총인구수"])
            top_age = int(ages[int(np.argmax(total_matrix[idx]))])
            c1, c2, c3 = st.columns(3)
            c1.metric("총인구", f"{total_pop:,}명")
            c2.metric("남성 인구", f"{int(m_vals.sum()):,}명")
            c3.metric("최다 인구 연령", f"{top_age}세")

# ---------------------------------------------------------
# TAB 2. 연령별 구조 비교 (여러 지역 겹쳐보기)
# ---------------------------------------------------------
with tab2:
    st.subheader("여러 지역의 연령별 인구 비율 비교")
    levels = ["시도", "시군구", "읍면동"]
    level2 = st.selectbox("행정구역 단위", levels, index=1, key="t2_level")
    filtered2 = meta[meta["레벨"] == level2]

    search2 = st.text_input("지역명 검색", key="t2_search")
    if search2:
        filtered2 = filtered2[filtered2["지역명"].str.contains(search2, case=False, na=False)]

    default_idx = filtered2.index[:2].tolist() if len(filtered2) >= 2 else filtered2.index.tolist()
    selected = st.multiselect(
        "비교할 지역 선택 (최대 8개)",
        options=filtered2.index.tolist(),
        default=default_idx,
        format_func=lambda i: meta.loc[i, "지역명"],
        key="t2_regions",
        max_selections=8,
    )

    if not selected:
        st.info("비교할 지역을 1개 이상 선택해주세요.")
    else:
        fig2 = go.Figure()
        for i in selected:
            pct = pct_matrix[i] * 100
            fig2.add_trace(go.Scatter(
                x=ages, y=pct, mode="lines", name=meta.loc[i, "지역명"],
                hovertemplate="%{x}세: %{y:.2f}%<extra>" + meta.loc[i, "지역명"] + "</extra>",
            ))
        fig2.update_layout(
            title="연령별 인구 비율(%) 비교",
            xaxis_title="연령",
            yaxis_title="전체 인구 대비 비율(%)",
            template="plotly_white",
            height=600,
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# TAB 3. 쌍둥이 지역 찾기
# ---------------------------------------------------------
with tab3:
    st.subheader("인구 구조가 가장 비슷한 '쌍둥이 지역' 찾기")
    st.caption("선택한 지역과 연령별 인구 비율(구조)이 가장 유사한 지역을 같은 행정구역 단위 안에서 찾아줍니다.")

    col_a, col_b = st.columns([1, 2])

    with col_a:
        idx3, level3 = region_picker(meta, key_prefix="t3")
        min_pop = st.slider(
            "비교 대상 최소 인구수 (너무 작은 지역 제외)",
            min_value=0, max_value=50000, value=1000, step=500,
            key="t3_minpop",
        )
        metric = st.radio(
            "유사도 계산 방식",
            ["유클리드 거리 (분포 차이가 작을수록 유사)", "코사인 유사도 (분포 모양이 비슷할수록 유사)"],
            key="t3_metric",
        )
        top_n = st.slider("표시할 쌍둥이 지역 수", 3, 15, 5, key="t3_topn")

    with col_b:
        if idx3 is not None:
            same_level = meta[(meta["레벨"] == level3) & (meta["총인구수"] >= min_pop)]
            candidate_idx = np.array([i for i in same_level.index if i != idx3])

            if len(candidate_idx) == 0:
                st.warning("조건에 맞는 비교 대상 지역이 없습니다. 최소 인구수를 낮춰보세요.")
            else:
                target_vec = pct_matrix[idx3]
                cand_vecs = pct_matrix[candidate_idx]

                if metric.startswith("유클리드"):
                    score = np.sqrt(((cand_vecs - target_vec) ** 2).sum(axis=1))
                    order = np.argsort(score)  # 작을수록 유사
                    score_label = "거리(작을수록 유사)"
                else:
                    denom = (np.linalg.norm(cand_vecs, axis=1) * np.linalg.norm(target_vec))
                    denom[denom == 0] = np.nan
                    score = (cand_vecs @ target_vec) / denom
                    order = np.argsort(-score)  # 클수록 유사
                    score_label = "코사인 유사도(클수록 유사)"

                top_idx = candidate_idx[order[:top_n]]
                top_score = score[order[:top_n]]

                result_df = pd.DataFrame({
                    "지역명": meta.loc[top_idx, "지역명"].values,
                    "총인구수": meta.loc[top_idx, "총인구수"].values,
                    score_label: top_score,
                })
                result_df["총인구수"] = result_df["총인구수"].map(lambda v: f"{int(v):,}")
                result_df[score_label] = result_df[score_label].round(4)

                target_name = meta.loc[idx3, "지역명"]
                st.markdown(f"**'{target_name}'** 과(와) 인구 구조가 가장 비슷한 지역 Top {top_n}")
                st.dataframe(result_df, use_container_width=True, hide_index=True)

                # 비교 차트: 대상 지역 vs 1위 쌍둥이 지역
                best_idx = top_idx[0]
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=ages, y=pct_matrix[idx3] * 100, mode="lines", name=target_name,
                    line=dict(width=3),
                ))
                fig3.add_trace(go.Scatter(
                    x=ages, y=pct_matrix[best_idx] * 100, mode="lines",
                    name=meta.loc[best_idx, "지역명"],
                    line=dict(width=3, dash="dash"),
                ))
                fig3.update_layout(
                    title=f"{target_name} vs {meta.loc[best_idx, '지역명']} — 연령별 인구 비율 비교",
                    xaxis_title="연령",
                    yaxis_title="전체 인구 대비 비율(%)",
                    template="plotly_white",
                    height=550,
                    hovermode="x unified",
                )
                st.plotly_chart(fig3, use_container_width=True)
