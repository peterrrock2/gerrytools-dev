from gerrytools.latex.seatsvotes import SeatsVotes


def test_latex_seatsvotes_generates_tikz_with_legend_entries():
    plot = SeatsVotes(include_legend=True, xlabel="Vote Share", ylabel="Seat Share", title="SV")
    plot.add_seat_votes_data(
        pov_party_vote_shares=[100, 200, 300, 400],
        total_vote_shares=[220, 390, 540, 700],
        name="Election A",
    )
    plot.add_proportionality_line()
    plot.add_efficiency_gap_line()
    plot.add_seat_votes_data(
        pov_party_vote_shares=[0.40, 0.52, 0.63, 0.57],
        name="Election B",
        linecolor="denim",
        markercolor="denim",
    )
    plot.add_custom_line(
        slope=0.5,
        linecolor="amber!20!denim",
        linestyle="-.",
        linewidth=1.5,
        label="Blend",
    )

    latex = str(plot)
    assert r"\begin{tikzpicture}" in latex
    assert r"\documentclass" not in latex
    assert r"\definecolor{" not in latex
    assert r"\color[HTML]{" in latex
    assert "HTML:" not in latex
    assert r"\color{amber!20!denim}" in latex
    assert r"\color{denim}" in latex
    assert "Election A" in latex
    assert "Election B" in latex
    assert "Blend" in latex
    assert "Efficiency Gap" in latex
    assert "Vote Share" in latex
    assert "Seat Share" in latex

    full_document = str(plot.document)
    assert r"\definecolor" in full_document


def test_latex_seatsvotes_hides_crosshairs_and_markers():
    plot = SeatsVotes(include_legend=False)
    plot.add_seat_votes_data(
        pov_party_vote_shares=[0.48, 0.52, 0.61, 0.44],
        name="Shares",
    )
    plot.remove_crosshairs()
    plot.hide_election_markers()

    latex = str(plot)
    assert "fill opacity=" not in latex
    assert "minimum size=" not in latex
