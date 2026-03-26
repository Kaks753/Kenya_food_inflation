# Quick push script for Food Inflation Tracker


echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🚀 PUSHING KENYA FOOD INFLATION TRACKER TO GITHUB"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📦 What will be pushed:"
echo "  • 11 files modified/created"
echo "  • All notebooks fixed and tested"
echo "  • Clean data generated (3 CSVs)"
echo "  • app.py corrected"
echo ""
echo "🔍 Current status:"
git log -1 --oneline
echo ""
echo "📊 Files changed:"
git diff --stat HEAD~1
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
read -p "Push to GitHub? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "🚀 Pushing to main branch..."
    git push origin main
    echo ""
    echo "✅ Push complete!"
    echo "🌐 View at: https://github.com/Kaks753/food-inflation"
else
    echo "❌ Push cancelled"
fi
