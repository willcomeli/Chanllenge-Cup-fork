import { ArrowRight, BriefcaseBusiness, CheckCircle2, FileCheck2, FileText, GraduationCap, LoaderCircle, PenLine, RotateCcw, ShieldCheck, Sparkles, UploadCloud, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Confidence, SectionHeader } from '../components/common'
import { AppLink } from '../router'
import { api } from '../services/api'
import type { ParsedResumeProfile } from '../types'

export default function ResumePage() {
  const [fileName, setFileName] = useState('')
  const [parsing, setParsing] = useState(false)
  const [parsed, setParsed] = useState(false)
  const [profile, setProfile] = useState<ParsedResumeProfile | null>(null)
  const [taskId, setTaskId] = useState('')
  const [parseError, setParseError] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const isLlmProfile = (nextProfile?: ParsedResumeProfile | null): nextProfile is ParsedResumeProfile => (
    nextProfile?.analysisSource === 'llm'
    && nextProfile.llmAnalysis?.enabled === true
    && nextProfile.llmAnalysis?.status === 'completed'
  )

  useEffect(() => {
    const savedTaskId = window.sessionStorage.getItem('latestResumeTaskId')
    if (savedTaskId === 'demo_resume_task') {
      window.sessionStorage.removeItem('latestResumeTaskId')
    } else if (savedTaskId) {
      api.getResumeTask(savedTaskId)
        .then((res) => {
          const nextProfile = res.data.result
          if (!isLlmProfile(nextProfile)) {
            window.sessionStorage.removeItem('latestResumeTaskId')
            return
          }
          setTaskId(savedTaskId)
          setProfile(nextProfile)
          setParsed(true)
        })
        .catch(() => window.sessionStorage.removeItem('latestResumeTaskId'))
    }
  }, [])

  const waitForResumeTask = async (id: string) => {
    const deadline = Date.now() + 5 * 60_000
    while (Date.now() < deadline) {
      const response = await api.getResumeTask(id)
      if (response.data.status === 'completed') return response
      if (response.data.status === 'failed') throw new Error(response.data.error || '简历解析失败')
      await new Promise((resolve) => window.setTimeout(resolve, 1_000))
    }
    throw new Error('简历解析超过 5 分钟，请稍后重试')
  }

  const parseResume = (file: File) => {
    setFileName(file.name)
    setParsing(true)
    setParsed(false)
    setParseError('')
    api.createResumeTask(file)
      .then((res) => {
        setTaskId(res.data.taskId)
        window.sessionStorage.setItem('latestResumeTaskId', res.data.taskId)
        return waitForResumeTask(res.data.taskId)
      })
      .then((res) => {
        setTaskId(res.data.taskId)
        const nextProfile = res.data.result
        if (!isLlmProfile(nextProfile)) {
          throw new Error('未得到 LLM 简历解析结果，请检查 LLM 配置后重新上传')
        }
        setProfile(nextProfile)
        setParsed(true)
      })
      .catch((error) => {
        setParseError(error instanceof Error ? error.message : '简历解析失败')
        setParsed(false)
      })
      .finally(() => {
        setParsing(false)
      })
  }

  const updateSkills = () => {
    if (!taskId || !profile) return
    api.updateResumeSkills(taskId, profile.skills)
      .then((res) => setProfile((current) => current ? ({ ...current, skills: res.data.skills }) : current))
      .catch(() => undefined)
  }

  return (
    <div className="page-stack resume-page">
      <section className="page-intro">
        <div><span className="section-eyebrow">AI RESUME PARSER</span><h2>把经历转化为可计算的能力画像</h2><p>支持 PDF、Word 简历解析，并将技能实体链接到岗位能力图谱。</p></div>
        <div className="privacy-chip"><ShieldCheck size={16} />文件仅用于本次分析</div>
      </section>

      <section className="resume-layout">
        <aside className="panel upload-panel">
          <SectionHeader eyebrow="STEP 01" title="上传简历" description="单个文件不超过 10 MB" />
          <button className={`dropzone ${parsing ? 'parsing' : ''}`} onClick={() => inputRef.current?.click()}>
            <input ref={inputRef} type="file" accept=".pdf,.doc,.docx,.txt,.png,.jpg,.jpeg,.webp" onChange={(event) => event.target.files?.[0] && parseResume(event.target.files[0])} />
            {parsing ? <><LoaderCircle size={34} className="spinner" /><strong>正在解析简历</strong><span>建立技能实体链接...</span></> : <><span className="upload-icon"><UploadCloud size={28} /></span><strong>拖拽文件到这里，或点击上传</strong><span>支持 PDF / DOC / DOCX</span></>}
          </button>
          {parseError && <p role="alert">{parseError}</p>}
          {fileName && <div className="file-item"><span><FileText size={19} /></span><div><strong>{fileName}</strong><small>1.8 MB · {parsed ? '解析完成' : '处理中'}</small></div>{parsed ? <CheckCircle2 size={18} className="success-icon" /> : <LoaderCircle size={18} className="spinner" />}</div>}
          <button className="ghost-button full-button" onClick={() => inputRef.current?.click()}><RotateCcw size={15} />重新上传</button>
        </aside>

        {parsed && profile ? <section className="panel resume-result visible">
          <div className="result-header"><div><span className="section-eyebrow">LLM PARSED PROFILE</span><h2>{profile.candidateName}的能力画像</h2><p>{profile.direction} · {profile.experienceYears} 年项目经验</p><span className="llm-chip"><Sparkles size={14} />{profile.llmAnalysis?.inputMode === 'vision' ? 'LLM 视觉解析' : 'LLM 文本解析'} · {profile.llmAnalysis?.model}</span></div><span className="parse-score"><small>解析完整度</small><strong>{profile.completeness}<em>%</em></strong></span></div>
          <div className="profile-summary">
            <div><span className="profile-avatar">{profile.candidateName.slice(0, 1)}</span><div><strong>{profile.candidateName}</strong><span>意向：{profile.targetPosition}</span></div></div>
            <span><GraduationCap size={17} />{profile.education}</span><span><BriefcaseBusiness size={17} />{profile.experienceYears} 年相关经验</span>
          </div>

          <div className="result-section">
            <div className="result-section-head"><h3>技能要素 <span>{profile.skills.length}</span></h3><button onClick={updateSkills}><PenLine size={14} />编辑修正</button></div>
            <div className="resume-skill-list">
              {profile.skills.map((skill) => (
                <div className="resume-skill" key={skill.name}><div><span className={`level-dot level-${skill.level}`} /><strong>{skill.name}</strong><em>{skill.level}</em></div><p>{skill.source}</p><Confidence value={skill.confidence} /></div>
              ))}
            </div>
          </div>

          <div className="result-section experience-section">
            <div className="result-section-head"><h3>核心经历 <span>{profile.experiences.length}</span></h3></div>
            <div className="experience-list">
              {profile.experiences.map((experience) => (
                <article key={experience.title}><i /><div><span>{experience.period}</span><h4>{experience.title}</h4><p>{experience.description}</p><div className="tag-list">{experience.skills.map((skill) => <em key={skill}>{skill}</em>)}</div></div></article>
              ))}
            </div>
          </div>

          <div className="result-footer"><span><FileCheck2 size={17} />已完成技能标准化与歧义消解</span><AppLink to="/match" className="primary-button">进入匹配诊断<ArrowRight size={16} /></AppLink></div>
        </section> : <section className="panel resume-result visible resume-empty-state">
          <span className="upload-icon"><FileText size={30} /></span>
          <h2>尚未生成能力画像</h2>
          <p>请先上传一份真实简历。解析完成后，这里将展示从简历中提取的技能与经历。</p>
          <button className="primary-button" onClick={() => inputRef.current?.click()}><UploadCloud size={16} />选择简历文件</button>
        </section>}
      </section>
    </div>
  )
}
