import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from pyod.models.knn import KNN
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import jdatetime
import openpyxl
st.title("📊 داشبورد دو فازی پروژه داده‌کاوی")
phase = st.sidebar.selectbox("انتخاب فاز پروژه:", ["فاز 1", "فاز 2"])

# ================== فاز 1 ==================
# ================== فاز 1 ==================
if phase == "فاز 1":
    st.header("📌 فاز 1: تحلیل اولیه")

    uploaded_file1 = st.file_uploader("آپلود فایل اکسل فاز 1", type=["xlsx"], key="phase1")
    if uploaded_file1:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.ensemble import IsolationForest
        from scipy.stats import zscore
        from pyod.models.knn import KNN

        # خواندن داده
        df = pd.read_excel(uploaded_file1)

        # پاکسازی داده‌ها
        df = df.dropna(subset=["تعداد مسافر", "مسافت کل", "نوع ناوگان", "شرکت مسافربری", "مبدا", "مقصد", "ساعت حرکت", "تاریخ حرکت"]).copy()
        df.loc[:, "ساعت حرکت"] = df["ساعت حرکت"].astype(str).str[:2].astype(int)

        # نمایش آمار
        st.subheader("نمایش آماری ویژگی‌ها")
        st.write(df.describe(include="all"))

        # نمودار 1
        sns.histplot(df["ساعت حرکت"], bins=10, kde=True)
        plt.title("توزیع ساعت حرکت")
        st.pyplot(plt)
        plt.close()

        # نمودار 2
        sns.boxplot(data=df, x="شرکت مسافربری", y="ساعت حرکت")
        plt.xticks(rotation=90)
        plt.title("پراکندگی ساعت حرکت بر اساس شرکت")
        st.pyplot(plt)
        plt.close()

        # نمودار 3
        group = df.groupby(["ساعت حرکت", "نوع ناوگان", "استان مقصد"]).size().reset_index(name="count")
        sns.scatterplot(data=group, x="ساعت حرکت", y="نوع ناوگان", hue="استان مقصد", size="count")
        plt.title("رابطه نوع ناوگان و ساعت حرکت بر اساس مقصد")
        st.pyplot(plt)
        plt.close()

        # نمودار 4
        group = df.groupby(["تعداد مسافر", "مسافت کل", "شرکت مسافربری"]).size().reset_index(name="count")
        sns.scatterplot(data=group, x="تعداد مسافر", y="مسافت کل", hue="شرکت مسافربری", size="count", alpha=0.7)
        plt.title("رابطه تعداد مسافر و مسافت کل بر شرکت مسافربری")
        st.pyplot(plt)
        plt.close()

        # رگرسیون خطی
        def extract_month(date_str):
            try:
                return int(date_str.split("/")[1])
            except:
                return 0
        df["ماه حرکت"] = df["تاریخ حرکت"].apply(extract_month)
        X = df[["مسافت کل", "نوع ناوگان", "شرکت مسافربری", "مبدا", "مقصد", "ساعت حرکت", "ماه حرکت"]]
        y = df["تعداد مسافر"]
        cat_cols = ["نوع ناوگان", "شرکت مسافربری", "مبدا", "مقصد"]
        preprocessor = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="passthrough")
        pipeline = Pipeline([("pre", preprocessor), ("model", LinearRegression())])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        st.write("MSE:", mean_squared_error(y_test, y_pred))
        st.write("R2 Score:", r2_score(y_test, y_pred))
        plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '-r')
        plt.xlabel("تعداد مسافر واقعی")
        plt.ylabel("تعداد مسافر پیش‌بینی‌شده")
        st.pyplot(plt)
        plt.close()

        errors = y_test - y_pred

        plt.figure(figsize=(8, 4))
        plt.hist(errors, bins=20, color='orange', edgecolor='black')
        plt.xlabel("مقدار خطا (واقعی - پیش‌بینی)")
        plt.ylabel("تعداد نمونه")
        plt.title("توزیع خطاها")
        plt.grid(True)
        plt.show()
        
        # طبقه‌بندی SVM
        def دسته_سفر(x):
            if x < 250:
                return 'کوتاه'
            elif x < 700:
                return 'متوسط'
            else:
                return 'بلند'
        df['نوع سفر'] = df['مسافت'].apply(دسته_سفر)
        X_cls = df[['تعداد مسافر', 'مسافت']].values
        le = LabelEncoder()
        y_cls = le.fit_transform(df['نوع سفر'])
        X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)
        model_linear = SVC(kernel='linear').fit(X_train, y_train)
        model_rbf = SVC(kernel='rbf').fit(X_train, y_train)

        for model, name in [(model_linear, "Linear"), (model_rbf, "RBF")]:
            y_pred = model.predict(X_test)
            st.write(f"{name} Accuracy:", accuracy_score(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        # خوشه‌بندی KMeans
        X_clust = df[['تعداد مسافر', 'مسافت']].values
        wcss = []
        for k in range(1, 5):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_clust)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 5), wcss, 'bo-')
        plt.title('Elbow Method')
        st.pyplot(plt)
        plt.close()

        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(X_clust)
        y_kmeans = kmeans.predict(X_clust)
        st.write(f"Silhouette Score: {silhouette_score(X_clust, y_kmeans):.3f}")
        sns.scatterplot(x=X_clust[:, 0], y=X_clust[:, 1], hue=y_kmeans, palette='Set2', s=100)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=200, marker='X')
        st.pyplot(plt)
        plt.close()

        # تشخیص ناهنجاری‌ها
        df["zscore"] = zscore(df["تعداد مسافر"])
        df["anomaly_z"] = df["zscore"].abs() > 3
        sns.scatterplot(x=df.index, y=df["تعداد مسافر"], hue=df["anomaly_z"], palette=["blue", "red"], s=20)
        st.pyplot(plt)
        plt.close()

        model_knn = KNN()
        model_knn.fit(df[["تعداد مسافر"]])
        df["anomaly_knn"] = model_knn.predict(df[["تعداد مسافر"]])
        sns.scatterplot(x=df.index, y=df["تعداد مسافر"], hue=df["anomaly_knn"], palette=["blue", "red"], s=20)
        st.pyplot(plt)
        plt.close()

        model_iso = IsolationForest(contamination=0.001, random_state=42)
        df["anomaly_iso"] = model_iso.fit_predict(df[["تعداد مسافر"]]) == -1
        sns.scatterplot(x=df.index, y=df["تعداد مسافر"], hue=df["anomaly_iso"], palette=["blue", "red"], s=20)
        st.pyplot(plt)
        plt.close()

        Q1 = df["تعداد مسافر"].quantile(0.25)
        Q3 = df["تعداد مسافر"].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df["anomaly_iqr"] = (df["تعداد مسافر"] < lower) | (df["تعداد مسافر"] > upper)
        sns.boxplot(x=df["تعداد مسافر"])
        st.pyplot(plt)
        plt.close()

# ================== فاز 2 ==================
# ================== فاز 2 ==================
elif phase == "فاز 2":
    st.header("📌 فاز 2: تحلیل پیشرفته")

    uploaded_file2 = st.file_uploader("آپلود فایل اکسل فاز 2", type=["xlsx"], key="phase2")
    if uploaded_file2:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from xgboost import XGBRegressor, XGBClassifier
        from sklearn.neural_network import MLPRegressor
        from sklearn.svm import SVC
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        import scipy.cluster.hierarchy as sch
        from mpl_toolkits.mplot3d import Axes3D
        import networkx as nx
        import jdatetime

        df = pd.read_excel(uploaded_file2)

        # پارت 1: تحلیل مسیرها و راننده‌ها
        df_clean = df.dropna(subset=["استان مبدا", "استان مقصد"]).copy()
        df_clean["مسیر"] = df_clean["استان مبدا"] + " → " + df_clean["استان مقصد"]
        top_routes = df_clean["مسیر"].value_counts().head(10)

        st.subheader("📊 10 مسیر پرتردد بین استان‌ها")
        sns.barplot(x=top_routes.values, y=top_routes.index, palette="viridis")
        st.pyplot(plt)
        plt.close()

        # رانندگان پرتردد
        drivers = pd.concat([
            df["راننده اول"].dropna(),
            df["راننده دوم"].dropna(),
            df["راننده سوم"].dropna()
        ])
        top_drivers = drivers.value_counts().head(10)
        st.subheader("👨‍✈️ ۱۰ راننده پرتردد")
        sns.barplot(x=top_drivers.values, y=top_drivers.index.astype(str), palette="crest")
        st.pyplot(plt)
        plt.close()

        # تبدیل تاریخ جلالی به میلادی
        def jalali_to_gregorian(jdate_str):
            try:
                y, m, d = map(int, jdate_str.split('/'))
                jd = jdatetime.date(y, m, d)
                return jd.togregorian()
            except:
                return None
        df["تاریخ میلادی"] = pd.to_datetime(df["تاریخ حرکت"].apply(jalali_to_gregorian))
        df["ماه"] = df["تاریخ میلادی"].dt.month
        df["روز هفته"] = df["تاریخ میلادی"].dt.dayofweek

        # پارت 2: مدل‌سازی رگرسیونی
        features = ["استان مبدا", "استان مقصد", "نوع ناوگان", "ساعت عددی", "شرکت مسافربری"]
        df["ساعت عددی"] = df["ساعت حرکت"].str.split(':').str[0].astype(float)
        df_model = df.dropna(subset=features + ["تعداد مسافر"])
        data = df_model[features + ["تعداد مسافر"]].copy()
        for col in ["استان مبدا", "استان مقصد", "نوع ناوگان", "شرکت مسافربری"]:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

        X = data[features]
        y = data["تعداد مسافر"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
            "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42)
        }

        st.subheader("📈 نتایج مدل‌های رگرسیونی")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write(f"**{name}**")
            st.write("MAE:", mean_absolute_error(y_test, y_pred))
            st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
            st.write("R2:", r2_score(y_test, y_pred))
            sns.scatterplot(x=y_test, y=y_pred)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            st.pyplot(plt)
            plt.close()

        # پارت 3: خوشه‌بندی
        df_cluster = df.dropna(subset=["تعداد مسافر", "مسافت", "ساعت عددی"]).copy()
        X_clust = df_cluster[["تعداد مسافر", "مسافت", "ساعت عددی"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clust)

        kmeans = KMeans(n_clusters=4, random_state=42)
        df_cluster["خوشه"] = kmeans.fit_predict(X_scaled)
        st.subheader("📦 خوشه‌بندی KMeans")
        sns.pairplot(df_cluster[["تعداد مسافر", "مسافت", "ساعت عددی", "خوشه"]], hue="خوشه", palette="tab10")
        st.pyplot(plt)
        plt.close()

        # پارت 4: تحلیل سری زمانی
        df_ts = df.dropna(subset=["تاریخ میلادی", "تعداد مسافر"]).copy()
        df_ts.set_index("تاریخ میلادی", inplace=True)
        weekly_passengers = df_ts["تعداد مسافر"].resample("W").sum()
        st.subheader("⏳ روند هفتگی تعداد مسافر")
        sns.lineplot(x=weekly_passengers.index, y=weekly_passengers.values, marker="o")
        st.pyplot(plt)
        plt.close()

        # پارت 5: نقشه حرارتی مسیرها
        heatmap_data = pd.crosstab(df['استان مبدا'], df['استان مقصد'])
        st.subheader("🌍 نقشه حرارتی تردد بین استان‌ها")
        plt.figure(figsize=(20, 8))
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False)
        st.pyplot(plt)
        plt.close()

        # پارت 6: تحلیل رانندگان
        df_driver = df[~df['راننده اول'].isna()].copy()
        driver_stats = df_driver.groupby('راننده اول').agg({'تعداد مسافر': ['sum', 'count']}).reset_index()
        driver_stats.columns = ['راننده', 'جمع مسافر', 'تعداد سفر']
        st.subheader("🚍 آمار رانندگان (راننده اول)")
        st.dataframe(driver_stats)
