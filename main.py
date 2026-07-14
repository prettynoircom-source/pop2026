import re
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(page_title="대한민국 연령별 인구 구조 탐색기", layout="wide")

DATA_FILE_HINT = "202606_202606_연령별인구현황_월간.csv"


# =========================================================
# 데이터 로딩 & 전처리
# =========================================================
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_raw():
    # 정확한 파일명이 없으면 폴더 안의 다른 csv도 탐색 (파일명이 매월 바뀌는 것에 대비)
    candidates = [DATA_FILE_HINT] + [f for f in glob.glob("*.csv") if f != DATA_FILE_HINT]
    for path in candidates:
        try:
            df = pd.read_csv(path, encoding="cp949", dtype=str)
            return df
        except FileNotFoundError:
            continue
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
            return df
    raise FileNotFoundError(
        "행정안전부 연령별 인구현황 CSV 파일을 찾을 수 없습니다. "
        "main.py와 같은 폴더에 CSV 파일을 두었는지 확인해주세요."
    )


def to_int(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("-", "0", regex=False)
        .replace("", "0")
        .astype(float)
        .fillna(0)
        .astype(int)
    )


@st.cache_data(show_spinner="인구 구조 데이터를 정리하는 중입니다...")
def preprocess(df: pd.DataFrame):
    df = df.copy()

    # ---- 행정구역명 / 행정코드 / 행정단위(레벨) 파싱 ----
    name_code = df["행정구역"].str.extract(r"^\s*(.+?)\s*\((\d+)\)\s*$")
    df["지역명"] = name_code[0].str.strip()
    df["행정코드"] = name_code[1]

    def get_level(code: str) -> str:
        if code[2:10] == "0" * 8:
            return "시도"
        elif code[5:10] == "0" * 5:
            return "시군구"
        else:
            return "읍면동"

    df["레벨"] = df["행정코드"].apply(get_level)

    # ---- 연령 컬럼 자동 탐지 (YYYY년MM월_성별_N세 / 100세 이상) ----
    age_pat = re.compile(r"^(\d{4}년\d{2}월)_(계|남|여)_(\d+)세$")
    age_pat_100 = re.compile(r"^(\d{4}년\d{2}월)_(계|남|여)_100세 이상$")

    prefix = None
    cols_by_gender = {"계": {}, "남": {}, "여": {}}
    for c in df.columns:
        m = age_pat.match(c)
        m100 = age_pat_100.match(c)
        if m:
            prefix, gender, age = m.group(1), m.group(2), int(m.group(3))
            cols_by_gender[gender][age] = c
        elif m100:
            prefix, gender = m100.group(1), m100.group(2)
            cols_by_gender[gender][100] = c

    if prefix is None:
        raise ValueError("연령별 인구 컬럼을 찾지 못했습니다. CSV 형식을 확인해주세요.")

    ages = list(range(0, 101))  # 0~99세 + 100세 이상

    # ---- 문자열(콤마) -> 정수 변환, 성별별 매트릭스 생성 ----
    matrices = {}
    for gender in ["계", "남", "여"]:
        cols = [cols_by_gender[gender][a] for a in ages]
        mat = df[cols].apply(to_int).to_numpy()
        matrices[gender] = mat

    total_col = [c for c in df.columns if c.endswith("_총인구수")][0]
    df["총인구수"] = to_int(df[total_col])

    meta = df[["지역명", "행정코드", "레벨", "총인구수"]].reset_index(drop=True)

    return meta, matrices, ages, prefix


# =========================================================
# 유사도(쌍둥이 지역) 계산
# =========================================================
def compute_age_ratio(mat_total: np.ndarray) -> np.ndarray:
    """행별(지역별) 연령 구성비(0~1) 매트릭스로 변환"""
    row_sum = mat_total.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return mat_total / row_sum


def find_twin_regions(idx: int, ratio_mat: np.ndarray, meta: pd.DataFrame,
                       same_level_only: bool, top_n: int = 10):
    level = meta.loc[idx, "레벨"]
    candidate_mask = np.ones(len(meta), dtype=bool)
    candidate_mask[idx] = False
    if same_level_only:
        candidate_mask &= (meta["레벨"] == level).to_numpy()
    candidate_mask &= (meta["총인구수"] > 0).to_numpy()

    target = ratio_mat[idx]
    diff = ratio_mat - target
    # 유클리드 거리 (연령 구성비 벡터 간 거리, 값이 작을수록 유사)
    dist = np.sqrt((diff ** 2).sum(axis=1))
    dist_masked = np.where(candidate_mask, dist, np.inf)

    order = np.argsort(dist_masked)[:top_n]
    result = meta.loc[order, ["지역명", "레벨", "총인구수"]].copy()
    result["유사도거리"] = dist_masked[order]
    result["유사도(%)"] = (1 - result["유사도거리"] / (result["유사도거리"].max() + 1e-9)) * 100
    return result.reset_index(drop=True)


# =========================================================
# 앱 본문
# =========================================================
def main():
    st.title("🇰🇷 대한민국 연령별 인구 구조 탐색기")
    st.caption("행정안전부 연령별 인구현황(월간) 데이터를 기반으로 한 인터랙티브 대시보드")

    raw = load_raw()
    meta, matrices, ages, prefix = preprocess(raw)
    ratio_mat = compute_age_ratio(matrices["계"])

    st.sidebar.header("🔎 지역 선택")
    level_filter = st.sidebar.radio("행정 단위", ["전체", "시도", "시군구", "읍면동"], horizontal=True)

    if level_filter == "전체":
        options_df = meta
    else:
        options_df = meta[meta["레벨"] == level_filter]

    search = st.sidebar.text_input("지역명 검색 (일부만 입력해도 됩니다)")
    if search:
        options_df = options_df[options_df["지역명"].str.contains(search, na=False)]

    if options_df.empty:
        st.sidebar.warning("검색 결과가 없습니다. 검색어를 다시 확인해주세요.")
        st.stop()

    region_name = st.sidebar.selectbox(
        "지역 선택",
        options_df["지역명"].tolist(),
        index=0,
    )
    region_idx = meta.index[meta["지역명"] == region_name][0]

    st.sidebar.markdown("---")
    st.sidebar.metric("선택 지역 총인구수", f"{meta.loc[region_idx, '총인구수']:,} 명")
    st.sidebar.caption(f"기준 시점: {prefix}")

    tab1, tab2, tab3 = st.tabs(["📊 인구 피라미드", "📈 연령별 분포 비교", "👯 쌍둥이 지역 찾기"])

    # ---------------------------------------------------
    # TAB 1. 인구 피라미드
    # ---------------------------------------------------
    with tab1:
        st.subheader(f"{region_name} 인구 피라미드")

        view_mode = st.radio("표시 방식", ["인구수(명)", "비율(%)"], horizontal=True, key="pyramid_mode")

        male = matrices["남"][region_idx].astype(float)
        female = matrices["여"][region_idx].astype(float)

        if view_mode == "비율(%)":
            total = male.sum() + female.sum()
            total = total if total > 0 else 1
            male_v = male / total * 100
            female_v = female / total * 100
            hover_suffix = "%"
        else:
            male_v = male
            female_v = female
            hover_suffix = "명"

        age_labels = [f"{a}세" if a < 100 else "100세 이상" for a in ages]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=age_labels, x=-male_v, orientation="h", name="남성",
            marker_color="#4C78A8",
            hovertemplate="나이: %{y}<br>남성: %{customdata:,.1f}" + hover_suffix + "<extra></extra>",
            customdata=male_v,
        ))
        fig.add_trace(go.Bar(
            y=age_labels, x=female_v, orientation="h", name="여성",
            marker_color="#F58518",
            hovertemplate="나이: %{y}<br>여성: %{customdata:,.1f}" + hover_suffix + "<extra></extra>",
            customdata=female_v,
        ))

        max_val = max(male_v.max(), female_v.max()) * 1.1
        fig.update_layout(
            barmode="relative",
            bargap=0.05,
            xaxis=dict(
                title=f"인구 ({hover_suffix})",
                range=[-max_val, max_val],
                tickvals=None,
            ),
            yaxis=dict(title="연령", categoryorder="array", categoryarray=age_labels),
            height=900,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="closest",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # TAB 2. 연령별 분포 비교 (여러 지역 겹쳐보기)
    # ---------------------------------------------------
    with tab2:
        st.subheader("연령별 인구 분포 비교")
        compare_options = options_df["지역명"].tolist()
        default_sel = [region_name] if region_name in compare_options else compare_options[:1]
        selected_regions = st.multiselect(
            "비교할 지역을 선택하세요 (여러 개 선택 가능)",
            options=meta["지역명"].tolist(),
            default=default_sel,
        )

        normalize = st.checkbox("연령 구성비(%)로 정규화해서 비교", value=True)

        if selected_regions:
            age_labels = [f"{a}세" if a < 100 else "100세 이상" for a in ages]
            fig2 = go.Figure()
            for r in selected_regions:
                ridx = meta.index[meta["지역명"] == r][0]
                total_vec = matrices["계"][ridx].astype(float)
                if normalize:
                    s = total_vec.sum()
                    y = total_vec / s * 100 if s > 0 else total_vec
                else:
                    y = total_vec
                fig2.add_trace(go.Scatter(
                    x=ages, y=y, mode="lines", name=r,
                    hovertemplate="나이: %{x}세<br>" + r + ": %{y:,.2f}<extra></extra>",
                ))
            fig2.update_layout(
                xaxis_title="연령",
                yaxis_title="구성비(%)" if normalize else "인구수(명)",
                height=600,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("비교할 지역을 1개 이상 선택해주세요.")

    # ---------------------------------------------------
    # TAB 3. 쌍둥이 지역 찾기
    # ---------------------------------------------------
    with tab3:
        st.subheader(f"'{region_name}'과(와) 인구 구조가 가장 비슷한 지역")
        st.caption("연령별 인구 구성비(전체 인구 대비 각 연령 비율)를 벡터로 비교해 유클리드 거리가 가장 가까운 지역을 찾습니다.")

        c1, c2 = st.columns([1, 1])
        with c1:
            same_level_only = st.checkbox("같은 행정 단위끼리만 비교 (예: 읍면동은 읍면동끼리)", value=True)
        with c2:
            top_n = st.slider("추천 지역 개수", min_value=3, max_value=30, value=10)

        twins = find_twin_regions(region_idx, ratio_mat, meta, same_level_only, top_n)

        st.dataframe(
            twins.rename(columns={"지역명": "지역", "레벨": "행정단위", "총인구수": "총인구수(명)"})
                 .style.format({"총인구수(명)": "{:,}", "유사도거리": "{:.5f}", "유사도(%)": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )

        if not twins.empty:
            best_twin = twins.iloc[0]["지역명"]
            st.success(f"🏆 가장 비슷한 지역: **{best_twin}** (1위)")

            twin_idx = meta.index[meta["지역명"] == best_twin][0]

            age_labels = [f"{a}세" if a < 100 else "100세 이상" for a in ages]
            base_vec = ratio_mat[region_idx] * 100
            twin_vec = ratio_mat[twin_idx] * 100

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=ages, y=base_vec, mode="lines", name=region_name, fill="tozeroy",
                line=dict(color="#4C78A8"),
                hovertemplate="나이: %{x}세<br>" + region_name + ": %{y:.2f}%<extra></extra>",
            ))
            fig3.add_trace(go.Scatter(
                x=ages, y=twin_vec, mode="lines", name=best_twin, fill="tozeroy",
                line=dict(color="#F58518"),
                opacity=0.6,
                hovertemplate="나이: %{x}세<br>" + best_twin + ": %{y:.2f}%<extra></extra>",
            ))
            fig3.update_layout(
                title=f"{region_name} vs {best_twin} 연령 구성비 비교",
                xaxis_title="연령",
                yaxis_title="구성비(%)",
                height=550,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
