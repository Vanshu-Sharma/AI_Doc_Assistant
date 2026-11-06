--
-- PostgreSQL database dump
--

\restrict Xp6zoZvfGEsyaxqXMqDtx2wEd5g8yrIjGd3MbTl3Vw5jDmQsMFef7mPjzGV2Uff

-- Dumped from database version 16.15 (Debian 16.15-1.pgdg13+2)
-- Dumped by pg_dump version 16.15 (Debian 16.15-1.pgdg13+2)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO postgres;

--
-- Name: document; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.document (
    id character varying NOT NULL,
    filename character varying NOT NULL,
    content_type character varying NOT NULL,
    size_bytes integer NOT NULL,
    uploaded_at timestamp with time zone NOT NULL,
    description character varying
);


ALTER TABLE public.document OWNER TO postgres;

--
-- Data for Name: alembic_version; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.alembic_version (version_num) FROM stdin;
465c4504adf5
\.


--
-- Data for Name: document; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.document (id, filename, content_type, size_bytes, uploaded_at, description) FROM stdin;
f12c3250-66b2-4786-b667-e1f5eab307af	sample_test.txt	text/plain	33	2026-10-29 10:08:05.043008+00	\N
bdd476b1-e1c1-4e2c-b93b-69e6dea40253	PostgreSQL and Docker Handbook.pdf	application/pdf	711901	2026-10-29 10:21:28.799302+00	\N
\.


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: document document_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.document
    ADD CONSTRAINT document_pkey PRIMARY KEY (id);


--
-- PostgreSQL database dump complete
--

\unrestrict Xp6zoZvfGEsyaxqXMqDtx2wEd5g8yrIjGd3MbTl3Vw5jDmQsMFef7mPjzGV2Uff

